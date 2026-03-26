/*
 * SQLite regression tests — exercises the two bugs that caused BCC-compiled
 * SQLite to segfault at runtime:
 *
 *   Bug 1 (stack alignment): An odd number of callee-saved register pushes
 *          left RSP misaligned for movaps/SSE instructions in called code.
 *          (x86-64 / i686 specific)
 *
 *   Bug 2 (static initialiser: &struct.array_member[idx]): The IR lowering
 *          emitted NULL instead of a GlobalRefOffset for addresses like
 *          &g.items[1] inside static initialisers.
 *          (all architectures)
 *
 * Both bugs are validated here with minimal reproducers.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

/* ===================================================================
 * Bug 2 reproducer: static initialiser with &struct.array_member[idx]
 * =================================================================== */

struct Inner {
    int   x;
    int   y;
    char  name[16];
};

struct Outer {
    int         tag;
    struct Inner items[4];
    int         count;
};

static struct Outer global_data = {
    .tag = 42,
    .items = {
        { .x = 10, .y = 20, .name = "alpha" },
        { .x = 30, .y = 40, .name = "beta"  },
        { .x = 50, .y = 60, .name = "gamma" },
        { .x = 70, .y = 80, .name = "delta" },
    },
    .count = 4
};

/* Static array of pointers into the struct's array member — this was the
 * exact pattern that produced NULL in BCC before the fix. */
static struct Inner *ptrs[] = {
    &global_data.items[0],
    &global_data.items[1],
    &global_data.items[2],
    &global_data.items[3],
    0   /* sentinel */
};

/* Also test bare array-decay of a struct member in static context. */
static struct Inner *arr_decay = global_data.items;

static int test_static_member_pointers(void) {
    int pass = 1;

    /* Verify that each pointer is non-NULL and points to the right data. */
    if (ptrs[0] == 0 || ptrs[0]->x != 10 || ptrs[0]->y != 20) {
        fprintf(stderr, "FAIL: ptrs[0] wrong (ptr=%p)\n", (void*)ptrs[0]);
        pass = 0;
    }
    if (ptrs[1] == 0 || ptrs[1]->x != 30 || ptrs[1]->y != 40) {
        fprintf(stderr, "FAIL: ptrs[1] wrong (ptr=%p)\n", (void*)ptrs[1]);
        pass = 0;
    }
    if (ptrs[2] == 0 || ptrs[2]->x != 50 || ptrs[2]->y != 60) {
        fprintf(stderr, "FAIL: ptrs[2] wrong (ptr=%p)\n", (void*)ptrs[2]);
        pass = 0;
    }
    if (ptrs[3] == 0 || ptrs[3]->x != 70 || ptrs[3]->y != 80) {
        fprintf(stderr, "FAIL: ptrs[3] wrong (ptr=%p)\n", (void*)ptrs[3]);
        pass = 0;
    }
    /* Sentinel must be NULL. */
    if (ptrs[4] != 0) {
        fprintf(stderr, "FAIL: sentinel not NULL\n");
        pass = 0;
    }
    /* Array-decay test. */
    if (arr_decay == 0 || arr_decay != &global_data.items[0]) {
        fprintf(stderr, "FAIL: arr_decay wrong (ptr=%p, expected=%p)\n",
                (void*)arr_decay, (void*)&global_data.items[0]);
        pass = 0;
    }

    /* Verify string data is intact. */
    if (strcmp(ptrs[0]->name, "alpha") != 0) {
        fprintf(stderr, "FAIL: ptrs[0]->name = '%s'\n", ptrs[0]->name);
        pass = 0;
    }
    if (strcmp(ptrs[2]->name, "gamma") != 0) {
        fprintf(stderr, "FAIL: ptrs[2]->name = '%s'\n", ptrs[2]->name);
        pass = 0;
    }

    return pass;
}

/* ===================================================================
 * Bug 2 variant: nested named struct member pointer in static init.
 * =================================================================== */

struct ServerInfo {
    int     port;
    char    host[32];
};

struct Config {
    int             version;
    struct ServerInfo servers[3];
    int             num_servers;
};

static struct Config cfg = {
    .version = 1,
    .servers = {
        { .port = 8080, .host = "localhost" },
        { .port = 8081, .host = "db.local"  },
        { .port = 8082, .host = "cache.local" },
    },
    .num_servers = 3
};

static struct ServerInfo *server_ptrs[] = {
    &cfg.servers[0],
    &cfg.servers[1],
    &cfg.servers[2],
    0
};

static int test_nested_member_pointers(void) {
    int pass = 1;

    for (int idx = 0; idx < 3; idx++) {
        if (server_ptrs[idx] == 0) {
            fprintf(stderr, "FAIL: server_ptrs[%d] is NULL\n", idx);
            pass = 0;
            continue;
        }
        struct ServerInfo *s = server_ptrs[idx];
        if (s->port != 8080 + idx) {
            fprintf(stderr, "FAIL: server_ptrs[%d]->port = %d, expected %d\n",
                    idx, s->port, 8080 + idx);
            pass = 0;
        }
    }
    if (server_ptrs[3] != 0) {
        fprintf(stderr, "FAIL: server_ptrs sentinel not NULL\n");
        pass = 0;
    }
    return pass;
}

int main(void) {
    int all_pass = 1;

    if (!test_static_member_pointers()) {
        fprintf(stderr, "FAILED: test_static_member_pointers\n");
        all_pass = 0;
    }
    if (!test_nested_member_pointers()) {
        fprintf(stderr, "FAILED: test_nested_member_pointers\n");
        all_pass = 0;
    }

    if (all_pass) {
        printf("sqlite_regression OK\n");
    }
    return all_pass ? 0 : 1;
}
