
def sum_to_n(arr, n):
    """
    takes an unordered array of unique integers and an integer n
    and returns all unique pairs which sum to n. via iteration.

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
        except_arr0 = arr[1:]
        for pair_of_arr0 in except_arr0:
            if pair_of_arr0 + arr[0] == n:
                result = tuple((arr[0], pair_of_arr0))
                except_arr0.remove(pair_of_arr0)
                break
        return [result] + sum_to_n(except_arr0, n)

a = [1, 2, 3, 4]
b = [1, 4, 5, 3, 2]
c = [1, 2, 5, 6, 3] 
n = 5
print(sum_to_n(a,n))


