


class info_extractor(): 
    def __init__(self):
        pass
    
    '''
        functions that extract the task-relevant part
    '''

    # main extract function
    def task_relevant_part_extract(self, args):
        '''
            args: extract_method
        '''
        pass

    # extract function that implements a specific method
    def _important_word_extraction_with_tfidf(self):
        pass

class style_transfer_agent(): # use gpt or bart or T5
    def __init__(self):

    '''
        functions that try to regenerate sentences given 
        task-relevant part and specific style
    '''
    def regenerate(self):
        pass

class victim_model():
    def __init__(self, args):
        '''
            args: model_name_or_path, 
        '''
        self.model
        self.tokenizer
