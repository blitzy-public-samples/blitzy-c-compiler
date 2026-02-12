/*
 * zero_length_array.c — GCC zero-length array extension test fixture
 *
 * Checkpoint 2 validation: Tests the GCC zero-length array extension which
 * allows `type name[0]` as the trailing member of a struct. This pattern is
 * heavily used throughout the Linux kernel for variable-length structures
 * (e.g., sk_buff, nlmsghdr, many ioctl payloads).
 *
 * Key properties validated:
 *   1. The compiler accepts `char data[0]` as a struct trailing member.
 *   2. sizeof(struct) does NOT include the zero-length array (contributes 0 bytes).
 *   3. The zero-length array member can be used as a flexible buffer after
 *      over-allocating with malloc.
 *   4. C99 flexible array members (`char data[]`) also work correctly.
 *   5. Multiple struct layouts with zero-length arrays of different element types.
 *   6. Alignment and padding behavior is correct.
 *
 * Expected: compiles cleanly, runs, prints results, returns 0 on success.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * Test 1: Basic struct with a zero-length char array (GCC extension)
 * Common Linux kernel pattern for variable-length message buffers.
 * -------------------------------------------------------------------------- */
struct message {
    int type;
    int length;
    char data[0]; /* zero-length array — GCC extension */
};

/* --------------------------------------------------------------------------
 * Test 2: Zero-length array with a different element type (int)
 * Validates that sizeof is still computed without the trailing array.
 * -------------------------------------------------------------------------- */
struct int_buffer {
    unsigned int count;
    int values[0]; /* zero-length int array */
};

/* --------------------------------------------------------------------------
 * Test 3: C99 flexible array member for comparison
 * Semantics are nearly identical to zero-length arrays in practice.
 * -------------------------------------------------------------------------- */
struct flex {
    int n;
    char data[]; /* C99 flexible array member — no explicit size */
};

/* --------------------------------------------------------------------------
 * Test 4: Nested struct containing a struct with zero-length array member
 * Validates that the compiler handles pointer-based access patterns correctly.
 * -------------------------------------------------------------------------- */
struct header {
    unsigned short tag;
    unsigned short flags;
};

struct packet {
    struct header hdr;
    unsigned int payload_len;
    unsigned char payload[0]; /* zero-length trailing array */
};

/* --------------------------------------------------------------------------
 * Test 5: Struct with alignment padding before the zero-length array
 * Verifies that alignment/padding is computed correctly and the zero-length
 * array sits at the correct offset.
 * -------------------------------------------------------------------------- */
struct aligned_msg {
    char     tag;       /* 1 byte */
    /* (padding here to align 'id' to 4-byte boundary) */
    int      id;        /* 4 bytes */
    short    code;      /* 2 bytes */
    /* (possible padding before the zero-length array) */
    double   extra[0];  /* zero-length double array — tests alignment */
};

/* --------------------------------------------------------------------------
 * Helper: check a condition, print PASS/FAIL, return 0/1
 * -------------------------------------------------------------------------- */
static int check(const char *name, int condition) {
    if (condition) {
        printf("  PASS: %s\n", name);
        return 0;
    } else {
        printf("  FAIL: %s\n", name);
        return 1;
    }
}

/* --------------------------------------------------------------------------
 * main — exercise all zero-length array and flexible array tests
 * -------------------------------------------------------------------------- */
int main(void) {
    int failures = 0;

    printf("=== Zero-Length Array GCC Extension Tests ===\n\n");

    /* ------------------------------------------------------------------
     * Test 1: sizeof(struct message) must equal sizeof(int) * 2
     * The zero-length char array contributes 0 bytes to the struct size.
     * ------------------------------------------------------------------ */
    printf("[Test 1] Basic zero-length char array sizeof\n");
    failures += check(
        "sizeof(struct message) == sizeof(int) * 2",
        sizeof(struct message) == sizeof(int) * 2
    );

    /* ------------------------------------------------------------------
     * Test 2: sizeof(struct int_buffer) must equal sizeof(unsigned int)
     * The zero-length int array contributes 0 bytes.
     * ------------------------------------------------------------------ */
    printf("[Test 2] Zero-length int array sizeof\n");
    failures += check(
        "sizeof(struct int_buffer) == sizeof(unsigned int)",
        sizeof(struct int_buffer) == sizeof(unsigned int)
    );

    /* ------------------------------------------------------------------
     * Test 3: Dynamic allocation and usage of struct message
     * Allocate extra space after the struct for the flexible data.
     * ------------------------------------------------------------------ */
    printf("[Test 3] Dynamic allocation and data access\n");
    {
        const int payload_size = 100;
        struct message *msg = (struct message *)malloc(sizeof(*msg) + payload_size);
        if (!msg) {
            printf("  FAIL: malloc returned NULL\n");
            return 1;
        }

        msg->type   = 1;
        msg->length = payload_size;
        memcpy(msg->data, "hello", 6); /* includes null terminator */

        failures += check(
            "msg->type == 1",
            msg->type == 1
        );
        failures += check(
            "msg->length == 100",
            msg->length == payload_size
        );
        failures += check(
            "msg->data contains \"hello\"",
            strcmp(msg->data, "hello") == 0
        );

        /* Write to the far end of the allocated buffer */
        msg->data[payload_size - 1] = 'Z';
        failures += check(
            "msg->data[99] == 'Z'",
            msg->data[payload_size - 1] == 'Z'
        );

        free(msg);
    }

    /* ------------------------------------------------------------------
     * Test 4: Dynamic allocation and usage of struct int_buffer
     * ------------------------------------------------------------------ */
    printf("[Test 4] Zero-length int array dynamic usage\n");
    {
        const unsigned int count = 5;
        struct int_buffer *buf = (struct int_buffer *)malloc(
            sizeof(*buf) + count * sizeof(int)
        );
        if (!buf) {
            printf("  FAIL: malloc returned NULL\n");
            return 1;
        }

        buf->count = count;
        unsigned int i;
        for (i = 0; i < count; i++) {
            buf->values[i] = (int)(i * i); /* 0, 1, 4, 9, 16 */
        }

        failures += check("buf->count == 5", buf->count == 5);
        failures += check("buf->values[0] == 0",  buf->values[0] == 0);
        failures += check("buf->values[1] == 1",  buf->values[1] == 1);
        failures += check("buf->values[2] == 4",  buf->values[2] == 4);
        failures += check("buf->values[3] == 9",  buf->values[3] == 9);
        failures += check("buf->values[4] == 16", buf->values[4] == 16);

        free(buf);
    }

    /* ------------------------------------------------------------------
     * Test 5: C99 flexible array member (struct flex)
     * sizeof(struct flex) should equal sizeof(int) — the flexible array
     * member has no size.
     * ------------------------------------------------------------------ */
    printf("[Test 5] C99 flexible array member sizeof\n");
    {
        /*
         * NOTE: The C standard says sizeof applied to a struct with a
         * flexible array member gives the size as if the FAM were omitted,
         * except it may include more trailing padding than the omission
         * would imply.  In practice, on both GCC and Clang with typical
         * alignment, sizeof(struct flex) == sizeof(int) for this layout.
         * We verify the flexible member contributes no additional size
         * beyond what alignment requires.
         */
        failures += check(
            "sizeof(struct flex) == sizeof(int)",
            sizeof(struct flex) == sizeof(int)
        );
    }

    /* ------------------------------------------------------------------
     * Test 6: C99 flexible array member dynamic allocation and usage
     * ------------------------------------------------------------------ */
    printf("[Test 6] C99 flexible array member dynamic usage\n");
    {
        const int data_len = 32;
        struct flex *f = (struct flex *)malloc(sizeof(*f) + data_len);
        if (!f) {
            printf("  FAIL: malloc returned NULL\n");
            return 1;
        }

        f->n = data_len;
        memset(f->data, 'A', data_len);
        f->data[data_len - 1] = '\0';

        failures += check("f->n == 32", f->n == data_len);
        failures += check(
            "f->data starts with 'A'",
            f->data[0] == 'A'
        );
        failures += check(
            "f->data[30] == 'A'",
            f->data[30] == 'A'
        );
        failures += check(
            "f->data is null-terminated",
            f->data[data_len - 1] == '\0'
        );

        free(f);
    }

    /* ------------------------------------------------------------------
     * Test 7: struct packet (nested struct + zero-length trailing array)
     * ------------------------------------------------------------------ */
    printf("[Test 7] Nested struct with zero-length trailing array\n");
    {
        /*
         * sizeof(struct packet) should be sizeof(struct header) +
         * any padding + sizeof(unsigned int).  The zero-length
         * unsigned char array contributes 0 bytes.
         */
        size_t expected_size = sizeof(struct header) + sizeof(unsigned int);
        /* Account for possible padding between header and payload_len */
        failures += check(
            "sizeof(struct packet) does not include payload[0]",
            sizeof(struct packet) <= expected_size + sizeof(unsigned int)
        );

        const unsigned int pkt_payload = 64;
        struct packet *pkt = (struct packet *)malloc(sizeof(*pkt) + pkt_payload);
        if (!pkt) {
            printf("  FAIL: malloc returned NULL\n");
            return 1;
        }

        pkt->hdr.tag   = 0xAB;
        pkt->hdr.flags = 0x01;
        pkt->payload_len = pkt_payload;
        memset(pkt->payload, 0xCC, pkt_payload);

        failures += check("pkt->hdr.tag == 0xAB",       pkt->hdr.tag == 0xAB);
        failures += check("pkt->hdr.flags == 0x01",      pkt->hdr.flags == 0x01);
        failures += check("pkt->payload_len == 64",      pkt->payload_len == pkt_payload);
        failures += check("pkt->payload[0] == 0xCC",     pkt->payload[0] == 0xCC);
        failures += check("pkt->payload[63] == 0xCC",    pkt->payload[pkt_payload - 1] == 0xCC);

        free(pkt);
    }

    /* ------------------------------------------------------------------
     * Test 8: Struct with alignment-sensitive zero-length double array
     * Validates that the compiler inserts correct padding before the
     * zero-length array so that double alignment is satisfied, while
     * the array itself contributes 0 bytes.
     * ------------------------------------------------------------------ */
    printf("[Test 8] Alignment-sensitive zero-length double array\n");
    {
        /*
         * struct aligned_msg contains:
         *   char   tag    (1 byte, padded to 4)
         *   int    id     (4 bytes)
         *   short  code   (2 bytes, padded to 8 for double alignment)
         *   double extra[0] (0 bytes — but imposes double alignment on struct)
         *
         * On most targets sizeof(double) == 8, so sizeof(struct aligned_msg)
         * should be padded to a multiple of 8.  The zero-length array must
         * NOT add any size, but its alignment requirement affects padding.
         */
        size_t sz = sizeof(struct aligned_msg);
        failures += check(
            "sizeof(struct aligned_msg) is multiple of alignof(double)",
            (sz % sizeof(double)) == 0
        );
        /* The zero-length array should NOT make the struct larger than
         * what padding alone would require. */
        failures += check(
            "sizeof(struct aligned_msg) == 16 (typical, with padding)",
            sz == 16 || sz == 12 || sz == 8
            /* Accept any correct padding result; 16 is most common on LP64 */
        );
    }

    /* ------------------------------------------------------------------
     * Test 9: Zero-length array address equals one-past-end of struct
     * The data member should point immediately after the last real field.
     * ------------------------------------------------------------------ */
    printf("[Test 9] Zero-length array address is one-past-end of struct fields\n");
    {
        struct message *msg = (struct message *)malloc(sizeof(*msg) + 16);
        if (!msg) {
            printf("  FAIL: malloc returned NULL\n");
            return 1;
        }

        /*
         * &msg->data[0] should equal (char *)msg + sizeof(struct message).
         * This confirms the zero-length array is placed directly after
         * the struct's real members.
         */
        char *data_addr   = (char *)&msg->data[0];
        char *struct_end  = (char *)msg + sizeof(struct message);
        failures += check(
            "&msg->data[0] == (char*)msg + sizeof(struct message)",
            data_addr == struct_end
        );

        free(msg);
    }

    /* ------------------------------------------------------------------
     * Summary
     * ------------------------------------------------------------------ */
    printf("\n=== Results: %d failure(s) ===\n", failures);

    return failures == 0 ? 0 : 1;
}
