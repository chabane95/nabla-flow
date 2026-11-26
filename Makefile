CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
LDLIBS ?= -lm
ifeq ($(OS),Windows_NT)
EXE = .exe
else
GPU_DL = -ldl
endif
SOURCES = src/main.c src/gpt.c src/data.c src/checkpoint.c src/recall.c src/scale.c
HEADERS = src/gpt.h src/data.h src/checkpoint.h
BIN = nablaflow$(EXE)

.PHONY: all check
all: $(BIN)

$(BIN): $(SOURCES) $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

check: $(BIN)
	./$(BIN) check

.PHONY: gpu
gpu: nfgpu$(EXE)

nfgpu$(EXE): tools/nfgpu.c src/gpu.c src/gpt.c src/checkpoint.c src/gpu.h src/gpt.h src/checkpoint.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc tools/nfgpu.c src/gpu.c src/gpt.c src/checkpoint.c $(LDFLAGS) $(LDLIBS) $(GPU_DL) -o $@
