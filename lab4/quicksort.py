def partition(coo, begin, end):
    col, row, data = coo
    pivot = (col[begin], row[begin])
    i = begin + 1
    j = end - 1

    while True:
        while i <= j and (col[i], row[i]) <= pivot:
            i = i + 1
        while i <= j and (col[j], row[j]) >= pivot:
            j = j - 1

        if i <= j:
            col[i], col[j] = col[j], col[i]
            row[i], row[j] = row[j], row[i]
            data[i], data[j] = data[j], data[i]
        else:
            col[begin], col[j] = col[j], col[begin]
            row[begin], row[j] = row[j], row[begin]
            data[begin], data[j] = data[j], data[begin]
            return j


def quicksort(coo, begin, end):
    """
    arguments:
    coo - (col, row, data) tuple that represents matrix in coo format
    begin,end - pretty straightforward - index where to start and where to finish sorting
    """
    if end - begin > 1:
        p = partition(coo, begin, end)
        quicksort(coo, begin, p)
        quicksort(coo, p + 1, end)
