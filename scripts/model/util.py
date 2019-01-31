def perf_measure_count(y_actual, y_predicted):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_predicted)):
        if y_actual[i]==y_predicted[i]==1:
           TP += 1
        if y_predicted[i]==1 and y_actual[i]!=y_predicted[i]:
           FP += 1
        if y_actual[i]==y_predicted[i]==0:
           TN += 1
        if y_predicted[i]==0 and y_actual[i]!=y_predicted[i]:
           FN += 1

    return TP, FP, TN, FN


def perf_measure(y_actual, y_hat):
    TP = []
    FP = []
    TN = []
    FN = []

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP.append(i)
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP.append(i)
        if y_actual[i]==y_hat[i]==0:
           TN.append(i)
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN.append(i)

    return TP, FP, TN, FN


# credit to https://stackoverflow.com/a/24542498/3197111
def full_print(*args, **kwargs):
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold='nan')
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)


def full_print_np_arr(arr):
    full_print(arr[:, None])
