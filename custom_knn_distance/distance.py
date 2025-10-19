import numpy as np

def gaussian_kernel(d):
    """ 
    Gaussian kernel fonksiyonu: uzaklığın karesi
    negatiflerin üssel fonksiyonuna dönüştürülerek
    yakın noktaların daha yüksek ağırlık almasını sağlar.
    
    """
    return np.exp(-d**2 / 2)

def D(y, X, r):
    """
    Yerel yoğunluk fonksiyonu D(y):
    y noktasının çevresindeki yoğunluğu ölçer.

    Parameters:
    - y: 1D numpy array (tek nokta)
    - X: 2D numpy array (tüm eğitim verisi)
    - r: float (arayüzde kullanılan yarıçap)

    Returns:
    - float: Normalize edilmiş yoğunluk değeri
    """
    K = lambda d: np.exp(-d**2 / 2)
    distances = np.linalg.norm(X - y, axis=1) / (r + 1e-12)
    return np.sum(gaussian_kernel(distances)) / (2 * np.pi * (r**2 + 1e-12))

def r(y, X):
    """
    Adaptif yarıçap fonksiyonu r(y):
    y noktası için yoğunluğa bağlı olarak dinamik yarıçap hesaplar.
    Yoğunluk arttıkça yarıçap küçülür.
    """
    initial_r = 1.0
    local_density = D(y, X, initial_r)
    return 1 / (1 + local_density)

def M(x, y, X):
    """
    Ağırlıklandırılmış mesafe fonksiyonu M(x, y):
    x ve y arasındaki fiziksel mesafeyi, y'nin yerel yoğunluğuna göre
    ters orantılı hale getirerek düzeltir.

    Returns:
    - float: Yoğunlukla ayarlanmış mesafe
    """
    adaptive_radius = r(y, X)
    local_density = D(y, X, adaptive_radius)
    return np.linalg.norm(x - y) / (local_density + 1e-12)

def create_custom_distance(X):
    def custom_distance(x, y):
        return M(x, y, X)
    return custom_distance