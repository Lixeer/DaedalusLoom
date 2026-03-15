#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/idf_additions.h"
#include "freertos/projdefs.h"

void app_main(void)
{   
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        printf("Hello World!\n");
        
        
    }
    
}
