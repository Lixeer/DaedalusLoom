#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static const char *TAG = "wifi_connect_wrap";

#define AP_SSID "Lixeer"
#define AP_PASSWORD "cwb20051027"

static void wifi_event_handler_AP(void *arg, esp_event_base_t event_base, int32_t event_id,
                                  void *event_data)
{
    // 处理AP模式下的客户端连接事件
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_AP_STACONNECTED)
    {
        // 获取客户端连接事件数据
        wifi_event_ap_staconnected_t *event = (wifi_event_ap_staconnected_t *)event_data;
        // 记录设备连接日志
        ESP_LOGI(TAG, "设备已连接! MAC: %02X:%02X:%02X:%02X:%02X:%02X", event->mac[0],
                 event->mac[1], event->mac[2], event->mac[3], event->mac[4], event->mac[5]);
    }
    // 处理AP模式下的客户端断开事件
    else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_AP_STADISCONNECTED)
    {
        // 获取客户端断开事件数据
        wifi_event_ap_stadisconnected_t *event = (wifi_event_ap_stadisconnected_t *)event_data;
        // 记录设备断开日志
        ESP_LOGI(TAG, "设备已断开. MAC: %02X:%02X:%02X:%02X:%02X:%02X", event->mac[0],
                 event->mac[1], event->mac[2], event->mac[3], event->mac[4], event->mac[5]);
    }
}

static void wifi_event_handler_STA(void *arg, esp_event_base_t event_base, int32_t event_id,
                                   void *event_data)
{
    // 处理WIFI事件
    printf("entry");
    if (event_base == WIFI_EVENT)
    {
        switch (event_id)
        {
        case WIFI_EVENT_STA_START:
            // 记录WIFI驱动就绪日志
            ESP_LOGI(TAG, "WiFi驱动就绪");
            break;
        case WIFI_EVENT_STA_CONNECTED:
            // 记录成功连接到热点日志
            ESP_LOGI(TAG, "已连接到热点");
            break;
        }
        // 处理IP事件
    }
    else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP)
    {
        // 获取IP事件数据
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        // 记录成功获取IP日志
        ESP_LOGI(TAG, "成功获取IP:" IPSTR, IP2STR(&event->ip_info.ip));
    }
}

void wifi_nonow_init()
{
    ESP_ERROR_CHECK(esp_netif_init());
    // 初始化事件循环
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    ESP_LOGI(TAG, "wifi_init_success");

    // 新增事件循环注册
    esp_event_handler_instance_t wifi_event_handle, ip_event_handle;
    // 注册WiFi事件处理器
    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler_STA, NULL,
                                        &wifi_event_handle);
    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler_AP, NULL,
                                        NULL);
    // 注册IP事件处理器（STA获取IP时触发）
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler_STA,
                                        NULL, &ip_event_handle);
    ESP_LOGI(TAG, "wifi_init_success");
    wifi_config_t ap_config = {
        .ap =
            {
                .ssid = AP_SSID,
                .password = AP_PASSWORD,
            },
    };
    esp_wifi_set_mode(WIFI_MODE_AP);
    esp_wifi_set_config(WIFI_IF_AP, &ap_config);
    ESP_LOGI(TAG, "wifi_AP_config");
    esp_wifi_start();
}
