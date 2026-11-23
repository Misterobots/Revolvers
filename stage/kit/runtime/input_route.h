#pragma once
#include <stdint.h>

// Opt-in, file-driven input diagnostics. Called only for game input, never host UI.
// Each player is sampled independently; replay uses relative 60 Hz game ticks.
int input_route_connected(void);
void input_route_poll(int player, uint64_t *buttons, int32_t axes[4], int blocked);
