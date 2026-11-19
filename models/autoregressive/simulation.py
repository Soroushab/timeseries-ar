
def simulate_ar(phi, initial_values, shocks):
    x = list(initial_values)

    for shock in shocks:
        ar_part = 0.0

        for j in range(len(phi)):
            ar_part += phi[j] * x[-(j + 1)]

        x.append(ar_part + shock)

    return x
