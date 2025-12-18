def array_mult(A, B):
    A_n = len(A) #baris
    # baris didapatkan dari berapa jumlah objek A
    
    A_m = len(A[0]) #kolom
    # kolom didapatkan dari ada berapa jumlah elemen yang ada di baris ke 0 
    # A = [[1, 2, 3], [3, 4, 5]]
    # A_n = 2
    # A_m = 3
    
    B_n = len(B) 
    B_m = len(B[0])
    
    assert A_m == B_n #menyatakan aturan matrix 
    # contoh:
    # matrix A dengan aturan 2x3 harus di dots berasama matrix b dengan aturan 3x2
    # baris x kolom
    # baris a x kolom b
    
    R_n = A_n #baris
    R_m = B_m #kolom
    # A: 2x3 (n x m) . B: 3x2 (n x m) = R: 2x2 (A_n x B_m)
    
    R = [[0 for j in range(R_m)] for i in range(R_n)]
    # diberikan nilai 0 untuk perulangan sementara sesuai dengan rentang kolom (R_m)
    # contoh:
    # R_m = 3, berari ada 3 kolom yang berisi 0 -> [0, 0, 0]
    
    # kemudian hasil perulangan sementara tersebut di loop kembali sesuai dengan rentang baris
    # [0, 0, 0] di loop sebanyak jumlah baris (R_n)
    # contoh:
    # R_n = 2, berati ada 2 baris yang berisi [0, 0, 0]
    
    # R = [[0, 0, 0], [0, 0, 0]]
    
    def row(M, r): return M[r] #mengembalikan berapa rows(baris) dari Matrix
    def col(M, c): return [v[c] for v in M] #mengebalikan berapa jumlah element cols(kolom) dari Matrix
    # kenapa harus menggunakan loop? 
    # karena python tidak menyimpan kolom
    # mengambil jumlah element(index) di setiap kolom ada berapa
    
    def dots(v1, v2): return sum([x*y for x,y in zip(v1, v2)])
    # dots product 2 vector 
    # perulangan yang di jumlahkan 
    # hanya beda di penulisan sytax dengan yang sebelumnya
    
    # zip = memasangkan element seposisi = v1[0] dengan v2[0] -> (v1[0], v2[0]) -> (x,y)
    # x*y = v1[0]*v2[0]
    
    # sebanyak ada berapa jumlah element seposisi v1 dan v2 kemudian di jumlahkan
    for i in range(R_n): # loop i di rentang hasil baris(R_n)
        # looping kedua untuk j di rentang hasil kolom(R_m)
        for j in range(R_m):
            R[i][j] = dots(row(A, i), col(B, j)) 
    return R
