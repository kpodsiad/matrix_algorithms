m = 600
n = 600
k = 600

A = reshape(collect(Float64, 1:m*k), (m, k))
B = reshape(collect(Float64, 1:k*n), (k, n))

function multVectAndScalar(a, b)
    a * b
end

function mul(A, B)
    m, k1 = size(A)
    k2, n = size(B)
    if k1 != k2
        error("cannot multiply matrixes")
    end
    C = reshape(zeros(Float64, m*n), (m, n))

    for p = 1:k1
        a = @view A[:,p]
        for j = 1:n
            b = B[p, j]
            C[:,j] += multVectAndScalar(a, b)
        end
    end
    C
end

function mul2(A, B)
    A * B
end

@time @views mul(A,B)
@time mul2(A, B)

