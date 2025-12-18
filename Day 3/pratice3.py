def two_lists(a, b):
    return [a[i]+b[i] for i in range(len(a))]

# Kita bikin wadah (fungsi) namanya two_lists, isinya ada dua objek a dan b.

# Nah fungsi ini bikin aturan:
# kita ambil elemen ke-i dari objek a lalu ditambah dengan elemen ke-i dari objek b.

# Nilai i ini mengikuti aturan dari range(len(a)), artinya:
# i berjalan dari 0 sampai indeks terakhir milik objek a.

# contoh:
print(two_lists([1,2,3], [4,5,6]))

# a = 1, 2, 3
# b = 4, 5, 6

#  len(i) = jumlah si a -> 3
#  range = ada berapa si index nya -> range(3) = 0 -> 1 -> 2

# jadi, 
# i = 0 -> a[0] + b[0] = 1 + 4 = 5
# i = 1 -> a[1] + b[1] = 2 + 5 = 7
# i = 2 -> a[2] + b[2] = 3 + 6 = 9

# jawabnnya [5, 7, 9]

def dots(v1, v2):
    return sum(v1[i]*v2[i] for i in range(len(v1)))

# untuk yang ini aturannya adalah kita mengambil elemen ke i di v1 
# lalu di kalikan ke elemen i yang di  namun karena ini summary 
# jadi semua elemennya akan dijumlah

# contohnya 
print(dots([1,2,3], [2,3,4]))

# v1 = 1, 2, 3
# v2 = 2, 3, 4

# (1x2) + (2x3) + (3x4)
# 2 + 6 + 12 = 20 -> jawabannya 

def add_n(a):
    def new_fun(b):
        return [x + a for x in b]
    return new_fun

# analogi sederhana:
# add_n = mesin pembuat stiker
# a = stiker
# new_fun = orang yang akan menempelkan stiker
# b = barang yang akan di tempel

# jadi, di fungsi new_fun ini dia mengikat si a
# fungsi ini mengembalikan nilai elemen x yang di dapat dari hasil input si b yang selalu di tambah nilai a
# kemudian mesin add_n akan mengembalikan fungsi new_fun

# contoh:
add_10 = add_n(10)

print(add_10([1,2,3]))

