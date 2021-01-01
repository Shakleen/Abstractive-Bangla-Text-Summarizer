import tensorflow as tf

generated = tf.constant([[1, 2, 3, 4, 5], [10, 11, 13, 15, 0]], dtype=tf.int32)
reference = tf.constant([[2, 3, 5, 7], [10, 11, 12, 14]], dtype=tf.int32)

count_1 = 0
count_2 = 0

for (gen, ref) in zip(generated, reference):
    for i in range(ref.shape[-1]):
        found_1st = tf.reduce_any(tf.where(gen == ref[i], True, False))

        if found_1st:
            count_1 += 1

            if i != ref.shape[-1]-1:
                found_2nd = tf.reduce_any(tf.where(gen == ref[i+1], True, False)) 

                if found_1st and found_2nd:
                    count_2 += 1
    
    def lcs(X, Y, m, n): 
        if m == 0 or n == 0: 
            return 0
        elif X[m-1] == Y[n-1]: 
            return 1 + lcs(X, Y, m-1, n-1)
        else: 
            return max(lcs(X, Y, m, n-1), lcs(X, Y, m-1, n))

    count_l = lcs(gen, ref, gen.shape[0], ref.shape[0])
    recall_l = count_l / ref.shape[0]
    precision_l = count_l / gen.shape[0]
    f1_l = 2 * precision_l * recall_l / (precision_l + recall_l)
    print(f"Rouge-l: P={precision_l} R={recall_l} F={f1_l}")

    recall_1 = count_1 / ref.shape[0]
    precision_1 = count_1 / gen.shape[0]
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
    print(f"Rouge-1: P={precision_1} R={recall_1} F={f1_1}")

    recall_2 = count_2 / ref.shape[0]
    precision_2 = count_2 / gen.shape[0]
    f1_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2)
    print(f"Rouge-2: P={precision_2} R={recall_2} F={f1_2}")