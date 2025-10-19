from .distance import M

def create_custom_distance(X_train):
    """
    Sklearn uyumlu özel mesafe fonksiyonu oluşturur.
    Bu fonksiyon, sorgu (x) ve eğitim noktası (y) arasındaki
    mesafeyi, y'nin bulunduğu bölgede yoğunluk bilgisine göre ayarlar.

    Parameters:
    - X_train: 2D numpy array, eğitim verisi (tüm örnekler)

    Returns:
    - callable: Sklearn ile uyumlu mesafe fonksiyonu
    """
    def custom_distance(x, y):
        return M(x, y, X_train)
    return custom_distance