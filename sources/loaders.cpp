#include"loaders.hpp"

// Each sample (message) is represented as a boolean for his class (only for the training set), and a vector of strings representing the words composing it
// with length >= min_size. Only alphanumeric characters can be part of a word. Every other character is considered as a delimiter.

void loaders::load_smileys(const std::string& path, std::vector<std::pair<std::string, bool>>& smileys){
    smileys.clear();
	 std::fstream in(path.c_str(), std::ios_base::in);
    std::string smiley;
    char x;
    while(in >> smiley){
        in >> x;
        smileys.push_back(std::make_pair(smiley, x=='+'));
    }
    in.close();
}

void loaders::load_corrections(const std::string& path, std::vector<std::pair<std::string, std::string>>& corrections){
    corrections.clear();
	 std::fstream in(path.c_str(), std::ios_base::in);
    std::string word, repl;
    while(in >> word){
        in >> repl;
        corrections.push_back(std::make_pair(word, repl));
    }
    in.close();
}

void loaders::getRegexps(std::vector<std::pair<std::string, std::string>>& replacements){
    replacements.clear();
    // Remove first character and comma
    replacements.push_back(std::make_pair("^[01],", " "));
    // Remove \\x** \x** \\n \n \\' \\\\n \\t \u**** \r \t \U******** and other backslashes
    replacements.push_back(std::make_pair("(\\\\x[a-f0-9]{2})|(\\\\n)|(\\\\t)|(\\\\u[0-9a-f]{4})|(\\\\r)|(\\\\U[0-9a-f]{8})"," "));
    replacements.push_back(std::make_pair("\\\\'"," "));   // Delicate area
    replacements.push_back(std::make_pair("\\\\",""));
    // Remove links
    replacements.push_back(std::make_pair("http://[^\\s]*",""));
    // Remove HTML tags
    replacements.push_back(std::make_pair("<[^>]*>(.*?)</[^>]*>"," \\1 "));
    //  Remove Other HTML tags, mostly bad ones.
    replacements.push_back(std::make_pair("<[^>]*>",""));
    // Remove punctuation
    replacements.push_back(std::make_pair("[\\^,+!\\?\\.\\:\\;\\&\\\"#%_~'=/\\-`@€\\|\\$\\(\\)><\\{\\}\\]\\[]"," "));
    // Remove numbers
    replacements.push_back(std::make_pair("[0-9]"," "));
    // Remove superfluous spaces
    replacements.push_back(std::make_pair("\\s+"," "));
    // Replace repeated characters
    replacements.push_back(std::make_pair("(.)\\1{2,}", "\\1\\1"));
}

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

void loaders::load_data(const std::string& pathTrain, 
                        const std::string& pathTest, 
                        std::vector<std::vector<std::string>>& trainText, 
                        std::vector<bool>& labels,
                        std::vector<std::vector<std::string>>& testText){
    // Resize vectors just to be sure.
    trainText.resize(loaders::train_size);
    testText.resize(loaders::test_size);
    labels.resize(loaders::train_size);
    
    // Tokens with which smileys are replaced.
    std::string smileyToken = " smiley ";
    std::string saddeyToken = " saddey ";
  
    // Load list of smileys and corrections from corresponding files.
    std::vector<std::pair<std::string, std::string>> corrections;
    std::vector<std::pair<std::string, bool>> smileys;
    load_smileys("../smileys", smileys);
    load_corrections("../corrections", corrections);
    
    // Prepare list of regex replacements to be made.
    std::vector<std::pair<std::string, std::string>> replacements;
    getRegexps(replacements);
    
    // Load train file for processing.
    std::fstream train_file(pathTrain.c_str(), std::ios_base::in);
    std::string sample;
    for(int line_cmp=0;getline(train_file,sample);line_cmp++){
        auto& finalSample = trainText[line_cmp];
		  labels[line_cmp] = sample[0] == '1';
        
        // Replace smileys with smiley/saddey token.
        for(std::pair<std::string, bool> p : smileys)
            sample = boost::regex_replace(sample, boost::regex(p.first), p.second ? smileyToken : saddeyToken);
        
        // Apply all regexps 
        for(std::pair<std::string, std::string> p : replacements){
            boost::regex r(p.first);
            std::string fmt = p.second;
            sample = boost::regex_replace(sample, r, fmt);
        }
        
        // Turn all to lowercase
        std::transform(sample.begin(), sample.end(), sample.begin(), ::tolower);
        
        // Correct auto-censored swear words
        for(std::pair<std::string, std::string> p : corrections)
            sample = boost::regex_replace(sample, boost::regex(p.first), p.second);
        
        // Remove excess *
        sample = boost::regex_replace(sample, boost::regex("\\*"), "");
        
        // Save to output vector.
        std::stringstream ss(sample);
        std::string word;
        while(ss >> word)
            finalSample.push_back(word);
    }
    train_file.close();
    
    // Load test file for processing.
	 std::fstream test_file(pathTest.c_str(), std::ios_base::in);
    for(int line_cmp=0;getline(test_file,sample);line_cmp++){
        auto& finalSample = testText[line_cmp];
        
        // Replace smileys with smiley/saddey token.
        for(std::pair<std::string, bool> p : smileys)
            sample = boost::regex_replace(sample, boost::regex(p.first), p.second ? smileyToken : saddeyToken);
        
        // Apply all regexps 
        for(std::pair<std::string, std::string> p : replacements){
            boost::regex r(p.first);
            std::string fmt = p.second;
            sample = boost::regex_replace(sample, r, fmt);
        }
        
        // Turn all to lowercase
        std::transform(sample.begin(), sample.end(), sample.begin(), ::tolower);
        
        // Correct auto-censored swear words
        for(std::pair<std::string, std::string> p : corrections)
            sample = boost::regex_replace(sample, boost::regex(p.first), p.second);
        
        // Remove excess *
        sample = boost::regex_replace(sample, boost::regex("\\*"), "");
        
        // Save to output vector.
        std::stringstream ss(sample);
        std::string word;
        while(ss >> word)
            finalSample.push_back(word);
    }
    test_file.close();
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