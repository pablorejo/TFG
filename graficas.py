"""In this file we go to see probabilitis of good clasify with more than 1 image
"""
import math

class Binomial():
    def __init__(self,N,P):
        self.N = N
        self.P = P
        self.MEDIA = N*P
        
    def calc_n_over_k(self,k):
        return math.factorial(self.N) / ( math.factorial(k) * math.factorial(self.N-k)) 
    def calc_binomial(self,k):
        result = self.calc_n_over_k(k) * (self.P**k) * ((1-self.P)**(self.N-k))
        # print(result)
        return result
    
class PError():
    
    def __init__(self, N, P_acierto, number_per_class):
        self.N = N
        self.P_acierto = P_acierto
        self.number_per_class = number_per_class
        self.binomial_especie = Binomial(   N,  P_acierto**4    * ((1-P_acierto)/(number_per_class-1)))
        self.binomial_genero = Binomial(    N,  P_acierto**3    * ((1-P_acierto)/(number_per_class-1)) * (1/number_per_class))
        self.binomial_familia = Binomial(   N,  P_acierto**2    * ((1-P_acierto)/(number_per_class-1)) * ((1/number_per_class)**2))
        self.binomial_orden = Binomial(     N,  P_acierto**1    * ((1-P_acierto)/(number_per_class-1)) * ((1/number_per_class)**3))
        self.binomial_clase = Binomial(     N,                    ((1-P_acierto)/(number_per_class-1)) * ((1/number_per_class)**4))
        
    def calc_p_error(self,k):
        print(self.binomial_especie.calc_binomial(k))
        k_error = self.binomial_especie.calc_binomial(k)     * (self.number_per_class -1) 
        k_error += self.binomial_genero.calc_binomial(k)     * (self.number_per_class -1) * self.number_per_class
        k_error += self.binomial_familia.calc_binomial(k)    * (self.number_per_class -1) * self.number_per_class **2
        k_error += self.binomial_orden.calc_binomial(k)      * (self.number_per_class -1) * self.number_per_class **3
        k_error += self.binomial_clase.calc_binomial(k)      * (self.number_per_class -1) * self.number_per_class **4
        return k_error
    
        
N = 5
P_acierto = 0.8
number_per_class = 4

binomial_acierto = Binomial(N,P_acierto)
print(binomial_acierto.calc_binomial(5))
k = math.ceil(binomial_acierto.MEDIA)

probabilidad_de_fallar = PError(N,P_acierto,number_per_class)

print(probabilidad_de_fallar.calc_p_error(1))
