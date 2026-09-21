
class AnalysisResults:
    print_each_analysis_time_increment = True
    total_analysis_time = 0.
    
    def __init__(self,initialize_empty_lists=[]):
        # Initialize empty lists
        for list_name in initialize_empty_lists:
            setattr(self, list_name, [])
                
    def add_to_analysis_time(self,tic,toc):
        self.total_analysis_time += toc - tic
        if self.print_each_analysis_time_increment:
            print(f"Adding {toc - tic:0.4f} seconds to total analysis run time.")

    def print_total_analysis_time(self):
        print(f"Total analysis time: {self.total_analysis_time:0.4f} seconds")

    def show_results(self):
        attributes = dir(self)
        for attr in attributes:
            if attr.startswith('__'):
                continue
            if attr in ['print_each_analysis_time_increment',
                        'add_to_analysis_time',
                        'print_total_analysis_time',
                        'show_results',
                        ]:
                continue
                
            print(attr)
