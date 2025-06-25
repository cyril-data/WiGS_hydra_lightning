# Summary: Inputs a long list of arguments and extracts only the arguments needed for a function. 
#          This function is used as different models and selector strategies use different arguments.
# Input:
#   Func: The function whose arguments will be filtered into.
#   ArgumentDictionary: A dictionary of arguments, usually SimulationConfigInput, whose specific 
#                       arguments which are needed for the function will be extracted.
# Output: 
#   FilteredArguments: Only the arguments needed for Func.

### Libraries ###
import inspect

def FilterArguments(Func, ArgumentDictionary):

    ### Set Up ###
    Signature = inspect.signature(Func)    
    FilteredArguments = {}
    
    ### Filter Arguments ###
    for ParameterName, _ in Signature.parameters.items():
        if ParameterName in ArgumentDictionary:
            FilteredArguments[ParameterName] = ArgumentDictionary[ParameterName]
    
    ### Return ###
    return FilteredArguments