import sys

def chunks(xs, n):
    n = max(1, n)
    return (xs[i:i+n] for i in range(0, len(xs), n))


results_file = sys.argv[1]

with open(results_file, 'r') as txt_file:
    lines = txt_file.readlines()

accs = [float(line.split(':')[1].strip()) for line in lines if line.startswith('acc')]

experiments = chunks(accs, 8)

for experiment in experiments:
    experiment = [experiment[idx] for idx in [0,1,4,5,2,3,6,7]]
    for acc in experiment:
        print('{}'.format(acc), end='\t')
    print('\n')
