
def sum_to_n(arr, n):
    """
    takes an unordered array of unique integers and an integer n
    and returns all unique pairs which sum to n.

    parameters:
        arr(list): the array of unique integers.
        n(int): the integer

    return:
        (list): all unique pairs which sum to n.
    """
    result = None
    if len(arr) < 2:
        return []
    else:
        temp_arr = arr[1:]
        for unique_integer in temp_arr:
            if unique_integer + arr[0] == n:
                result = tuple((arr[0], unique_integer))
                temp_arr.remove(unique_integer)
                break
        return [result] + sum_to_n(temp_arr, n)
a = 'a'
print(a)
