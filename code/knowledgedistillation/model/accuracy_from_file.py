def is_all_same(candidate,reference):
    if len(candidate)!=len(reference):
        return False

    ans = True
    for i in range(len(candidate)):
        if candidate[i]!=reference[i]:
            ans = False
            break

    return ans


def post_process(candidate):
    new_candidate = []
    c_left = 0
    c_right = 0
    for token in candidate:
        new_candidate.append(token)
        if token == '(':
            c_left = c_left +1
        elif token == ')':
            c_right = c_right +1


    if c_right > c_left:
        for j in range(c_right - c_left):
            new_candidate.pop()

    if c_right < c_left:
        for j in range(c_left - c_right):
            new_candidate.append(')')

    return new_candidate

groundtruth = "test_f.txt"
predicted  = "output.txt"

g = []
p = []
f=open(groundtruth)
for line in f:
	g.append(line.strip().split())


f=open(predicted)
for line in f:
        p.append(line.strip().split())


count = 0
for i in range(len(g)):
	output_processed = post_process(p[i])
	if is_all_same(output_processed, g[i]):
		count += 1

print float(count)/len(g)
