#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>


typedef struct {{ struct_name }} {
    uint8_t *data;
    uint64_t len;
    char *error;
    uint32_t error_len;
} {{ struct_name }};

#ifdef __cplusplus__
extern "C" {
#endif

struct {{ struct_name }} {{ function_name }}(const uint8_t *input_ptr, uint64_t input_len);
void {{ free_function }}(struct {{ struct_name }} *_self);

#ifdef __cplusplus__
}
#endif