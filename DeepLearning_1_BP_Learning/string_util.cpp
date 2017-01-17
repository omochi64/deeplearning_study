#include <string>
#include <vector>
#include "string_util.h"

void string_split(const std::string &str, char split_char, std::vector<std::string> &out)
{
	int start = 0;
	int cur = 0;
	out.clear();

	if (str.length() == 0) return;

	while (start < str.length() && cur < str.length())
	{
		if (str[cur] == split_char)
		{
			out.push_back(str.substr(start, cur - start));
			start = cur + 1;
		}
		cur++;
	}

	// ÅŒã‚ª split_char ‚¶‚á–³‚¢ê‡A‚Ü‚¾ push ‚µ‚Ä‚¢‚È‚¢‚Ì‚ªc‚Á‚Ä‚¢‚é‚Ì‚Å“ü‚ê‚é
	if (str[str.length() - 1] != split_char)
	{
		out.push_back(str.substr(start, cur - start));
	}
}

std::string string_rstrip(const std::string &str)
{
	int strip_count = 0;
	while (str.length() > strip_count)
	{
		char last = str[str.length() - 1 - strip_count];
		if (last == ' ' || last == '\n' || last == '\r')
		{
			strip_count++;
		}
		else
		{
			break;
		}
	}

	return str.substr(0, str.length() - strip_count);
}

