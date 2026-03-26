/*
 * tests/fixtures/zero_length_array.c
 *
 * Checkpoint 2 — GCC Zero-Length Array Extension Test
 *
 * Validates that the compiler correctly handles zero-length arrays as
 * flexible array-like struct members. This is a pre-C99 GCC extension
 * (distinct from C99 flexible array members which use []) that is still
 * widely used in the Linux kernel for variable-length data patterns.
 *
 * Key properties tested:
 *   - sizeof(struct) does NOT include the zero-length array member
 *   - Elements can be written/read through the zero-length array tail
 *   - The pattern works with dynamic allocation of extra trailing space
 *   - Multiple struct layouts with zero-length arrays of different types
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Classic Linux kernel pattern: header + variable-length data */
struct packet {
    int length;
    int type;
    char data[0];  /* GCC zero-length array */
};

/* Second struct to test with a different element type */
struct int_buffer {
    unsigned int count;
    int values[0];  /* GCC zero-length array of ints */
};

/* Struct with zero-length array and alignment considerations */
struct aligned_msg {
    char tag;
    int id;
    double payload[0];  /* GCC zero-length array of doubles */
};

int main(void) {
    /* ===== Test 1: sizeof should not include the zero-length array ===== */
    if (sizeof(struct packet) != 2 * sizeof(int)) {
        printf("FAIL: sizeof(packet) = %zu, expected %zu\n",
               sizeof(struct packet), 2 * sizeof(int));
        return 1;
    }

    /* ===== Test 2: Allocate with extra space for char data ===== */
    int data_len = 10;
    struct packet *pkt = malloc(sizeof(struct packet) + data_len);
    if (!pkt) { printf("FAIL: malloc\n"); return 1; }

    pkt->length = data_len;
    pkt->type = 1;

    /* Write through the zero-length array */
    for (int i = 0; i < data_len; i++) {
        pkt->data[i] = (char)(i + 'A');
    }

    /* Read back and verify */
    for (int i = 0; i < data_len; i++) {
        if (pkt->data[i] != (char)(i + 'A')) {
            printf("FAIL: data[%d] = %c, expected %c\n",
                   i, pkt->data[i], (char)(i + 'A'));
            free(pkt);
            return 1;
        }
    }

    free(pkt);

    /* ===== Test 3: sizeof for int zero-length array struct ===== */
    if (sizeof(struct int_buffer) != sizeof(unsigned int)) {
        printf("FAIL: sizeof(int_buffer) = %zu, expected %zu\n",
               sizeof(struct int_buffer), sizeof(unsigned int));
        return 1;
    }

    /* ===== Test 4: Allocate and use int zero-length array ===== */
    int num_values = 5;
    struct int_buffer *buf = malloc(sizeof(struct int_buffer) + num_values * sizeof(int));
    if (!buf) { printf("FAIL: malloc int_buffer\n"); return 1; }

    buf->count = num_values;
    for (int i = 0; i < num_values; i++) {
        buf->values[i] = (i + 1) * 100;
    }

    /* Verify int array contents */
    for (int i = 0; i < num_values; i++) {
        int expected = (i + 1) * 100;
        if (buf->values[i] != expected) {
            printf("FAIL: values[%d] = %d, expected %d\n",
                   i, buf->values[i], expected);
            free(buf);
            return 1;
        }
    }

    free(buf);

    /* ===== Test 5: memcpy through zero-length array ===== */
    const char *msg = "HelloWorld";
    int msg_len = 10;
    struct packet *pkt2 = malloc(sizeof(struct packet) + msg_len);
    if (!pkt2) { printf("FAIL: malloc pkt2\n"); return 1; }

    pkt2->length = msg_len;
    pkt2->type = 2;
    memcpy(pkt2->data, msg, msg_len);

    /* Verify memcpy'd data */
    if (memcmp(pkt2->data, msg, msg_len) != 0) {
        printf("FAIL: memcpy through zero-length array\n");
        free(pkt2);
        return 1;
    }

    free(pkt2);

    /* ===== Test 6: aligned_msg struct with double zero-length array ===== */
    /* sizeof should account for padding but NOT the zero-length array */
    struct aligned_msg *amsg = malloc(sizeof(struct aligned_msg) + 3 * sizeof(double));
    if (!amsg) { printf("FAIL: malloc aligned_msg\n"); return 1; }

    amsg->tag = 'X';
    amsg->id = 42;
    amsg->payload[0] = 1.5;
    amsg->payload[1] = 2.5;
    amsg->payload[2] = 3.5;

    if (amsg->payload[0] != 1.5 || amsg->payload[1] != 2.5 || amsg->payload[2] != 3.5) {
        printf("FAIL: aligned_msg payload mismatch\n");
        free(amsg);
        return 1;
    }

    free(amsg);

    printf("zero_length_array OK\n");
    return 0;
}
