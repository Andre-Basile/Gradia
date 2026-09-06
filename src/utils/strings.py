
def search_substr_in_list(string_array, substr):
    for index, string in enumerate(string_array):
        if substr in string:return (True, index)
    return (False, None)