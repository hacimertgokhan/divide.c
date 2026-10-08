# Doğruluğu artırırken hız koşulunu koruma

Varsayılan model artık sıralamayı ayıran döndürmeli birleştirme ve sorguyla çarpımlı çıktı okuması kullanır. Parametre sayısı 1.112, attention bloğu sayısı 11, boyut 12 ve yakın token sayısı 8 olarak korunmuştur.

## Değişikliklerin gerekçesi

Eski ortalama aynı çiftteki sol/sağ vektörlerinin yerini ayırt edemiyordu. Sağ vektörün boyutlarını bir konum döndürmek bu simetriyi ekstra matris çarpımı olmadan kırar. Bu kayıpsız veya tüm sıralamayı koruyan bir kodlama değildir.

Eski `context + query` çıktı okumasını izleyen doğrusal çıktı katmanında, sorgu sıkıştırılmış bir bloğun farklı bileşenlerini koşullu olarak seçemiyordu. Yeni `context + context * query` okuması bileşen bazında etkileşim sağlar; 12 çarpım ekler, yeni parametre eklemez.

Yaprak kodlayıcı konumdan bağımsızdır. Aynı token kimliğinin kodlaması aynı ileri hesapta paylaşılır; böylece tekrar eden tokenler ve sorgu için aynı matris çarpımı/tanh yeniden yapılmaz. Bu cache her örnekte sıfırlanır; optimizer değişiminden sonra eski temsiller kullanılmaz. Normalize attention ağırlıkları da boyutlar arasında paylaşılır. Bu optimizasyonlar eski ve normal mimariye de uygulanabilir; yalnızca Divide'a ait bir hız üstünlüğü olarak yorumlanmamalıdır.

## Ortak doğruluk karşılaştırması

64 karakter bağlam, 20.000 eğitim örneği/adımı, 4.000 ortak değerlendirme örneği. Seed yalnızca başlangıç ağırlıklarını değiştirir; eğitim ve test veri tohumları sabittir. Her satırda aynı başlangıç ve eğitim verisi karşılaştırılmıştır.

| Başlangıç tohumu | Eski Divide | Yeni Divide | Artış (yüzde puan) |
|---|---:|---:|---:|
| 42 | %30,925 | %43,075 | 12,150 |
| 43 | %35,875 | %46,800 | 10,925 |
| 44 | %30,150 | %44,675 | 14,525 |
| Ortalama | %32,317 | %44,850 | 12,533 |

Seed 42 yeni CSV'si ilk 2.000 örnek için %43,7 gösterir; tabloda hız aracının ortak 4.000 örnekteki %43,075 sonucu kullanılmıştır. Eski mimarinin control koşuları da yeni hesap paylaşımı optimizasyonlarıyla çalışır. Seed 42 eski checkpointi orijinal motorla değerlendirildiğinde aynı %30,925 doğruluk ve 1,915112 NLL elde edilir.

Seed 42 ara deneyleri: yalnız sıralamayı ayırma %28,85 (2.000 test); yalnız çarpımlı okuma %27,0 (4.000 test). İkisinin birlikte kullanılması bu koşuda daha iyi çalıştı. Farklı test sayıları nedeniyle bu ara sonuçları doğrudan bir istatistiksel ablation tablosu olarak değerlendirmemek gerekir.

## Hız kabul koşulu

`speed_check.py` orijinal motoru `original.c` anlık görüntüsünden, yeni motoru güncel `divide.c` kaynağından aynı -O3 ile derler. Aynı seed 42 ağırlıkları, aynı 4.000 test örneği ve aynı tek CPU süreci düzeni kullanılır. Sıkıştırmayı yeniden oluşturma ve autodiff grafiği dahil edilir. Her süreçte bir ısınma geçişi ve dokuz süre ölçümü vardır. Eski/yeni süreçler iki kez dönüşümlü çalıştırılır. Sonuç, iki süreç medyanının medyanıdır.

| Ölçüm | Orijinal Divide | Yeni Divide |
|---|---:|---:|
| Örnek başına süre | 0,124408 ms | 0,078247 ms |
| Doğruluk | %30,925 | %43,075 |
| NLL | 1,915112 | 1,729312 |
| Eğitim grafiği tepe düğüm belleği | 1.990.000 B | 1.311.720 B |

Ölçülen süre %37,10 azaldı. `speed_gate_passed` ve `accuracy_gate_passed` true. Bellek satırı kullanılan autodiff düğümlerinin boyutudur; toplam RAM veya ayrılmış arena kapasitesi değildir. Zamanlar sistem yükü nedeniyle değişken: eski süreç medyanları 0,112313 ve 0,136502 ms; yeni süreç medyanları 0,072773 ve 0,083721 ms. JSON dosyası tüm ölçümleri içerir.

Bu sonuçlar sabit konumlu anahtar/sayı eşleşmeleri ve tek bağlam uzunluğundaki küçük sentetik görev içindir. Eşleşme sıralaması, yerleri veya uzunluğu değişen metinlere, gerçek dil modellemeye ya da GPU sistemlerine genellenmiş hız/doğruluk iddiası değildir. Yeni doğruluk hâlâ normal attention'ın bu görevdeki yaklaşık %68–70 düzeyinin altında.

## Kullanım

```sh
make check
./divide --steps 20000 --tests 4000 --seed 42 --save new-model --csv new-results.csv
python3 experiments/speed_check.py
```

Bu çalışma sırasında seed 42 ile eğitilmiş yeni ağırlıklar kökte `improved.divide.weights` ve `improved.baseline.weights` olarak da kaydedilmiştir. Orijinal `model.*.weights` korunmuştur. Tahminde DIVIDE3 dosyasındaki sıkıştırıcı/okuma seçimi otomatik yüklenir; normal model için `--baseline` seçilir.

## Doğrulama

- Eski/yeni okuma ve sıkıştırıcının tüm kombinasyonlarında iki mimarinin seçilmiş parametreleri sonlu fark gradyanlarıyla doğrulandı; tekrar eden tokenlerin paylaşılan embeddingleri de kapsandı.
- Yeni birleştiricinin sol/sağ değişimine farklı tepki verdiği, eski birleştiricinin simetrik olduğu kontrol edildi.
- Eğitim, kaydetme/yükleme ve tahmin akışı çalıştırıldı; tarihi DIVIDE1 checkpointi de yüklendi.
- AddressSanitizer ve UndefinedBehaviorSanitizer ile kontroller geçti.
