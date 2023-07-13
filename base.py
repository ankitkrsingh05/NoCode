class NormalityTestsBase:
    def __init__(self, data):
        self.data = data
    
    def perform_test(self):
        raise NotImplementedError
    
    def display_results(self):
        raise NotImplementedError