#pragma once

#define BLOCK_SIZE 32
#define WIDTH 1920
#define HEIGHT 1080
#define NX 512
#define NY 512
#define DOMAIN_LENGTH 1.0f
#define DX ((DOMAIN_LENGTH) / ((float)((NX) - 1)))
#define DY ((DOMAIN_LENGTH) / ((float)((NY) - 1)))