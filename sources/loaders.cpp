#include"loaders.hpp"

// Each sample (message) is represented as a boolean for his class (only for the training set), and a vector of strings representing the words composing it
// with length >= min_size. Only alphanumeric characters can be part of a word. Every other character is considered as a delimiter.

void loaders::load_train(const std::string& path,std::vector<std::vector<std::string> >& text,std::vector<bool>& labels)
{
	text.resize(loaders::train_size);
	labels.resize(loaders::train_size);
	
	std::fstream train_file(path.c_str(),std::fstream::in);
	std::string line;
	// Getting each line (ie each message) one at the time.
	for(int line_cmp=0;getline(train_file,line);line_cmp++)
	{
		int line_size = line.size();
		auto& sample = text[line_cmp];
		// The class of the sample is determinated by his first character only.
		labels[line_cmp] = (line[0] == '1');
		// Since each line begins by [01],""" we can start the analysis only at the 6-th character.
		int word_start = 5,cur_position = 5;
		// While we're not at the end of the line, we consider each word one at the time.
		while(cur_position < line_size)
		{
			// While we're not at the end of the word (ie we didn't meet a delimter), the cursor advances.
			while(cur_position < line_size && (isalpha(line[cur_position]) || isdigit(line[cur_position])))
				++cur_position;
			
			// We add the word to sample if it's long enough.
			if(cur_position - word_start >= loaders::min_size)
				sample.push_back(line.substr(word_start,cur_position - word_start));
			
			++cur_position;
			word_start = cur_position;
		}
	}
}

void loaders::load_test(const std::string& path,std::vector<std::vector<std::string> >& text)
{
	text.resize(loaders::test_size);
	
	std::fstream train_file(path.c_str(),std::fstream::in);
	std::string line;
	// Getting each line (ie each message) one at the time.
	for(int line_cmp=0;getline(train_file,line);line_cmp++)
	{
		int line_size = line.size();
		auto& sample = text[line_cmp];
		// Since each line begins by """ we can start the analysis only at the 4-th character.
		int word_start = 3,cur_position = 3;
		// While we're not at the end of the line, we consider each word one at the time.
		while(cur_position < line_size)
		{
			// While we're not at the end of the word (ie we didn't meet a delimter), the cursor advances.
			while(cur_position < line_size && (isalpha(line[cur_position]) || isdigit(line[cur_position])))
				++cur_position;
			
			// We add the word to sample if it's long enough.
			if(cur_position - word_start >= loaders::min_size)
				sample.push_back(line.substr(word_start,cur_position - word_start));
			
			++cur_position;
			word_start = cur_position;
		}
	}
}