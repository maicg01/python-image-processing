A = ["p", "a", "a", "a", "p", "a"]
B = [1, 2, 3, 4, 5, 6]
count_a_removed = 0

# Lặp ngược qua A để tránh vấn đề khi xóa phần tử
for i in range(len(A) - 1, -1, -1):
    if A[i] == "a" and count_a_removed < 3:
        del A[i]
        del B[i]
        count_a_removed += 1

print("List A sau khi xóa:", A)
print("List B sau khi xóa:", B)