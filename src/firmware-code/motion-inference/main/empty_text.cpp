#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "driver/gpio.h"
#include "driver/sdmmc_host.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "sd_pwr_ctrl_by_on_chip_ldo.h"
#include "sdmmc_cmd.h"

static const char *TAG = "cnn1d_sd";

static const char *FW_MARKER = "sdcard-cnn1d-20260605";
static const char *SD_MOUNT_POINT = "/sdcard";
static const char *MODEL_PATH = "/sdcard/cnn1d_model.espdl";
static const char *INPUT_PATH = "/sdcard/cnn1d_input.txt";

static constexpr gpio_num_t SD_PWR_EN_GPIO = GPIO_NUM_45;
static constexpr int SDMMC_IO_LDO_CHAN_ID = 4;
static constexpr size_t CNN1D_INPUT_CHANNELS = 114;
static constexpr size_t CNN1D_INPUT_LENGTH = 32;
static constexpr size_t CNN1D_INPUT_SIZE = CNN1D_INPUT_CHANNELS * CNN1D_INPUT_LENGTH;
static constexpr size_t CNN1D_OUTPUT_CLASSES = 3;

static void log_heap_status(const char *stage)
{
    ESP_LOGI(TAG,
             "%s heap: internal_free=%u internal_largest=%u default_largest=%u psram_largest=%u simd_largest=%u",
             stage,
             static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL)),
             static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL)),
             static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_DEFAULT)),
             static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM)),
             static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_SIMD)));
}

static esp_err_t enable_sdcard_power(void)
{
    gpio_config_t io_conf = {
        .pin_bit_mask = 1ULL << SD_PWR_EN_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
#if SOC_GPIO_SUPPORT_PIN_HYS_FILTER
        .hys_ctrl_mode = GPIO_HYS_SOFT_DISABLE,
#endif
    };
    esp_err_t ret = gpio_config(&io_conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to configure SD power gpio: %s", esp_err_to_name(ret));
        return ret;
    }

    // ESP32-P4-Function-EV-Board uses an active-low SD_PWRn enable on GPIO45.
    ret = gpio_set_level(SD_PWR_EN_GPIO, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to enable SD power: %s", esp_err_to_name(ret));
    }
    return ret;
}

static esp_err_t mount_sdcard(sdmmc_card_t **out_card)
{
    if (!out_card) {
        return ESP_ERR_INVALID_ARG;
    }

    esp_err_t ret = enable_sdcard_power();
    if (ret != ESP_OK) {
        return ret;
    }

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 4,
        .allocation_unit_size = 16 * 1024,
        .disk_status_check_enable = false,
        .use_one_fat = false,
    };

    sd_pwr_ctrl_ldo_config_t ldo_config = {
        .ldo_chan_id = SDMMC_IO_LDO_CHAN_ID,
    };
    sd_pwr_ctrl_handle_t pwr_ctrl_handle = nullptr;
    ret = sd_pwr_ctrl_new_on_chip_ldo(&ldo_config, &pwr_ctrl_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "failed to create SDMMC IO LDO power control: %s", esp_err_to_name(ret));
        return ret;
    }

    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.pwr_ctrl_handle = pwr_ctrl_handle;
    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();
    slot_config.width = 4;
    slot_config.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    ESP_LOGI(TAG, "mounting SD card at %s", SD_MOUNT_POINT);
    ret = esp_vfs_fat_sdmmc_mount(SD_MOUNT_POINT, &host, &slot_config, &mount_config, out_card);
    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "failed to mount FAT filesystem on SD card");
        } else {
            ESP_LOGE(TAG, "failed to initialize SD card: %s", esp_err_to_name(ret));
        }
        sd_pwr_ctrl_del_on_chip_ldo(pwr_ctrl_handle);
        return ret;
    }

    sdmmc_card_print_info(stdout, *out_card);
    return ESP_OK;
}

static int8_t quantize_i8(float value, int exponent)
{
    const float inv_scale = std::pow(2.0f, static_cast<float>(-exponent));
    float q = std::round(value * inv_scale);
    if (q > 127.0f) {
        q = 127.0f;
    } else if (q < -128.0f) {
        q = -128.0f;
    }
    return static_cast<int8_t>(q);
}

static int16_t quantize_i16(float value, int exponent)
{
    const float inv_scale = std::pow(2.0f, static_cast<float>(-exponent));
    float q = std::round(value * inv_scale);
    if (q > 32767.0f) {
        q = 32767.0f;
    } else if (q < -32768.0f) {
        q = -32768.0f;
    }
    return static_cast<int16_t>(q);
}

static bool set_model_input(dl::TensorBase *input, const float *values, size_t count)
{
    if (!input || !values || input->get_size() < count) {
        ESP_LOGE(TAG, "input tensor is invalid");
        return false;
    }

    const int exponent = input->get_exponent();
    switch (input->get_dtype()) {
    case dl::DATA_TYPE_FLOAT: {
        float *dst = input->get_element_ptr<float>();
        for (size_t i = 0; i < count; ++i) {
            dst[i] = values[i];
        }
        return true;
    }
    case dl::DATA_TYPE_INT8: {
        int8_t *dst = input->get_element_ptr<int8_t>();
        for (size_t i = 0; i < count; ++i) {
            dst[i] = quantize_i8(values[i], exponent);
        }
        return true;
    }
    case dl::DATA_TYPE_INT16: {
        int16_t *dst = input->get_element_ptr<int16_t>();
        for (size_t i = 0; i < count; ++i) {
            dst[i] = quantize_i16(values[i], exponent);
        }
        return true;
    }
    default:
        ESP_LOGE(TAG, "unsupported input dtype: %s", input->get_dtype_string());
        return false;
    }
}

static bool read_logits(dl::TensorBase *output, float *logits, size_t count)
{
    if (!output || !logits || output->get_size() < count) {
        ESP_LOGE(TAG, "output tensor is invalid");
        return false;
    }

    const int exponent = output->get_exponent();
    const float scale = std::pow(2.0f, static_cast<float>(exponent));
    switch (output->get_dtype()) {
    case dl::DATA_TYPE_FLOAT: {
        const float *src = output->get_element_ptr<float>();
        for (size_t i = 0; i < count; ++i) {
            logits[i] = src[i];
        }
        return true;
    }
    case dl::DATA_TYPE_INT8: {
        const int8_t *src = output->get_element_ptr<int8_t>();
        for (size_t i = 0; i < count; ++i) {
            logits[i] = static_cast<float>(src[i]) * scale;
        }
        return true;
    }
    case dl::DATA_TYPE_INT16: {
        const int16_t *src = output->get_element_ptr<int16_t>();
        for (size_t i = 0; i < count; ++i) {
            logits[i] = static_cast<float>(src[i]) * scale;
        }
        return true;
    }
    default:
        ESP_LOGE(TAG, "unsupported output dtype: %s", output->get_dtype_string());
        return false;
    }
}

static int argmax(const float *logits, size_t count)
{
    int best = 0;
    for (size_t i = 1; i < count; ++i) {
        if (logits[i] > logits[best]) {
            best = static_cast<int>(i);
        }
    }
    return best;
}

static const char *class_name(int cls)
{
    switch (cls) {
    case 0:
        return "C0";
    case 1:
        return "C1";
    case 2:
        return "C2";
    default:
        return "UNKNOWN";
    }
}

static void consume_input_delimiters(FILE *file)
{
    int c = EOF;
    while ((c = std::fgetc(file)) != EOF) {
        if (c == ',' || c == ';' || c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            continue;
        }
        std::ungetc(c, file);
        break;
    }
}

static bool read_cnn1d_input_file(const char *path, float *values, size_t count)
{
    FILE *file = std::fopen(path, "r");
    if (!file) {
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        if (std::fscanf(file, " %f", &values[i]) != 1) {
            ESP_LOGE(TAG, "failed to read input value %u from %s", static_cast<unsigned>(i), path);
            std::fclose(file);
            return false;
        }
        consume_input_delimiters(file);
    }

    consume_input_delimiters(file);
    const int extra = std::fgetc(file);
    if (extra != EOF) {
        ESP_LOGW(TAG, "%s has extra data after %u values; extra data ignored",
                 path, static_cast<unsigned>(count));
    }

    std::fclose(file);
    ESP_LOGI(TAG, "loaded %u float values from %s", static_cast<unsigned>(count), path);
    return true;
}

static bool file_exists(const char *path)
{
    FILE *file = std::fopen(path, "rb");
    if (!file) {
        return false;
    }
    std::fclose(file);
    return true;
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "firmware marker: %s", FW_MARKER);
    log_heap_status("before SD mount");

    sdmmc_card_t *card = nullptr;
    if (mount_sdcard(&card) != ESP_OK) {
        return;
    }

    FILE *model_file = std::fopen(MODEL_PATH, "rb");
    if (!model_file) {
        ESP_LOGE(TAG, "model file not found: %s", MODEL_PATH);
        esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
        return;
    }

    std::fseek(model_file, 0, SEEK_END);
    const long model_size = std::ftell(model_file);
    std::fclose(model_file);
    if (model_size <= 0) {
        ESP_LOGE(TAG, "model file is empty or unreadable: %s", MODEL_PATH);
        esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
        return;
    }

    ESP_LOGI(TAG, "loading model from %s, size=%ld bytes", MODEL_PATH, model_size);
    log_heap_status("before model load");

    dl::Model model(MODEL_PATH,
                    fbs::MODEL_LOCATION_IN_SDCARD,
                    16 * 1024,
                    dl::MEMORY_MANAGER_GREEDY,
                    nullptr,
                    false);
    log_heap_status("after model load");

    dl::TensorBase *input = model.get_input();
    dl::TensorBase *output = model.get_output();
    if (!input || !output) {
        ESP_LOGE(TAG, "failed to get model input/output tensors");
        esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
        return;
    }

    ESP_LOGI(TAG, "input dtype=%s size=%d exponent=%d", input->get_dtype_string(), input->get_size(), input->get_exponent());
    ESP_LOGI(TAG, "output dtype=%s size=%d exponent=%d", output->get_dtype_string(), output->get_size(), output->get_exponent());
    if (input->get_size() < CNN1D_INPUT_SIZE || output->get_size() < CNN1D_OUTPUT_CLASSES) {
        ESP_LOGE(TAG, "unexpected tensor sizes: input=%d expected=%u output=%d expected=%u",
                 input->get_size(), static_cast<unsigned>(CNN1D_INPUT_SIZE),
                 output->get_size(), static_cast<unsigned>(CNN1D_OUTPUT_CLASSES));
        esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
        return;
    }

    float *input_values = static_cast<float *>(heap_caps_malloc(CNN1D_INPUT_SIZE * sizeof(float),
                                                               MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!input_values) {
        input_values = static_cast<float *>(heap_caps_malloc(CNN1D_INPUT_SIZE * sizeof(float), MALLOC_CAP_8BIT));
    }
    if (!input_values) {
        ESP_LOGE(TAG, "failed to allocate input buffer for %u floats", static_cast<unsigned>(CNN1D_INPUT_SIZE));
        esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
        return;
    }

    if (file_exists(INPUT_PATH)) {
        if (!read_cnn1d_input_file(INPUT_PATH, input_values, CNN1D_INPUT_SIZE)) {
            std::free(input_values);
            esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
            return;
        }
    } else {
        ESP_LOGW(TAG, "input file not found: %s; using zero-filled test sample", INPUT_PATH);
        for (size_t i = 0; i < CNN1D_INPUT_SIZE; ++i) {
            input_values[i] = 0.0f;
        }
    }

    if (!set_model_input(input, input_values, CNN1D_INPUT_SIZE)) {
        std::free(input_values);
        esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
        return;
    }

    model.run();

    float logits[CNN1D_OUTPUT_CLASSES] = {0.0f, 0.0f, 0.0f};
    if (!read_logits(output, logits, CNN1D_OUTPUT_CLASSES)) {
        std::free(input_values);
        esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
        return;
    }

    const int cls = argmax(logits, CNN1D_OUTPUT_CLASSES);
    printf("cnn1d input=[1,%u,%u] -> class=%d label=%s logits=[%.6f, %.6f, %.6f]\n",
           static_cast<unsigned>(CNN1D_INPUT_CHANNELS),
           static_cast<unsigned>(CNN1D_INPUT_LENGTH),
           cls,
           class_name(cls),
           logits[0],
           logits[1],
           logits[2]);

    std::free(input_values);
    esp_vfs_fat_sdcard_unmount(SD_MOUNT_POINT, card);
}
