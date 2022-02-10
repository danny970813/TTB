st = 'abcde'
if 'abc' in st:
    print('Y')
else:
    print('N')

for i in range(5):
    print(i)

a= 3

b = 2

print(a/b)

from pathlib import Path
root = Path('boat_dataset')
if root.exists():
    print('YES')
else:
    print('NO')