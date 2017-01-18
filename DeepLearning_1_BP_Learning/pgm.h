#include <cmath>
#include <string>

class PGM
{
public:
	static inline float clamp(float x) {
		if (x < 0.0)
			return 0.0;
		if (x > 1.0)
			return 1.0;
		return x;
	}

	static inline int to_int(float x) {
		return int(x * 255 + 0.5);
	}

	static void save_pgm_file(const std::string &filename, const float *gray_image, const int width, const int height) {
		FILE *f;
		fopen_s(&f, filename.c_str(), "wb");
		fprintf(f, "P2\n%d %d\n%d\n", width, height, 255);
		for (int i = 0; i < width * height; i++)
		{
			fprintf(f, "%d%s", to_int(gray_image[i]), (i+1)%width == 0 ? "\n" : " ");
		}
		fclose(f);
	}
};
