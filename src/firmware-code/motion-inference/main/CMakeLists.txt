idf_component_register(
    SRCS "empty_text.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp_driver_gpio esp_driver_sdmmc fatfs sdmmc
)
