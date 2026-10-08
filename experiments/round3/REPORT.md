# Üçüncü tur: yöntemi koruyarak eğitim ve hesap paylaşımını iyileştirme

Yakın tokenler ayrıntılı, uzak geçmiş ikili bloklar halinde kalıyor. Varsayılan sıkıştırıcı associative-selective, çıktı okuması product, ilişki kodlaması additive olarak korundu. Seçilen iki değişiklik eğitim dağılımı ve matematiksel olarak aynı önem hesabının paylaşılmasıdır. Boyut 12, parametre sayısı 1.112 ve 64 token için attention bloğu sayısı 11.

## Başarılı değişiklikler

### Eğitimde aynı sayıların tekrarını kaldırma

Eğitimde dört anahtarın değerleri farklı seçiliyor (--distinct yes). Her örnek yine dört anahtar/sayı, dolgu ve tek sorgudan oluşuyor. Öğrenme oranı, örnek bütçesi 100.000, optimizer ve model korunuyor. Değerlendirmede bu kısıt kapatılıyor; önceki bağımsız sayı seçimi ve tekrar eden sayılar aynen bulunuyor. Model çıkarımında kural tabanlı cevap, sayı ayrıştırıcısı veya özel tahmin yolu eklenmedi.

Dört bağımsız değer arasında en sık sayıyı seçip sorguyu tamamen yok sayan bir değerlendirme baseline'ının beklenen doğruluğu %38,35; ayrı veride ölçülen %38,875. Bazı başarısız yöntemlerin %37–39 civarında takılması bununla uyumlu. Bu benzerlik tek başına onların tam olarak bu kuralı öğrendiğini kanıtlamaz. Farklı değerli eğitim, bu kolay stratejinin doğruluğunu %25'e indirerek ilişkiye odaklanmayı teşvik eden bir veri düzeni denemesidir.

### Tekrarlanan önem hesabını paylaşma

Birleştirici her çocuğun önemini score(h)=w·h/sqrt(D) ile hesaplıyor. Aynı token çiftinin temsili ve puanı aynı ileri hesapta paylaşılabiliyor. Ebeveyn vektörü sqrt(2)*(wa*left + wb*right) olduğundan doğrusallıkla:

```text
score(parent) = sqrt(2) * (wa*score(left) + wb*score(right))
```

Böylece her seviyede tekrar D bileşenli önem çarpımı yapılmıyor. Bir skaler çalışma istatistiği saklanıyor; yeni parametre veya attention slotu yok. Sabit ağırlıklarda bütün parametre gradyanlarının referansla eşitliği test edildi. Kayıp ve tahmin eşitliği ayrıca eski motorla gerçek veride ölçüldü. Kayan noktalı gradyan toplama sırası uzun eğitimde küçük farklılıklara yol açabilir; cache ile eğitim de üç tohumda ayrıca çalıştırıldı.

## Ayrı değerlendirme

Model seçimi geliştirme tohumu 987654321 ile yapıldı. Son karşılaştırma daha önce bu turda kullanılmayan 700003 tohumunda, her düzen için 4.000 örnek ve üç başlangıç tohumu 42/43/44 ile yapıldı. Eski ağırlıklar ikinci turdaki identity-product-mixed modelleridir. Yeni ağırlıklar distinct-42/43/44 modelleridir. Her iki taraf 100.000 örnek ve 100.000 optimizer güncellemesi kullanır; batch=1. Değerlendirme dağılımı ve örnekler iki taraf için aynıdır. Eğitim dağılımı seçilen değişiklik nedeniyle farklıdır.

| Düzen | Önceki önerilen yöntem | Yeni eğitim |
|---|---:|---:|
| fixed | %66.900 | %74.967 |
| shuffled | %66.967 | %74.058 |
| scattered | %71.775 | %81.308 |
| mixed | %69.183 | %78.025 |

Karma düzen, tek tek başlangıç tohumları:

| Seed | Önceki | Yeni |
|---|---:|---:|
| 42 | %69.875 | %83.650 |
| 43 | %62.650 | %72.700 |
| 44 | %75.025 | %77.725 |

Yeni önerilen model seed=42 modelidir; en iyi tohum seçilmedi. Ortalama artış yaklaşık 8,84 yüzde puan. Bu sentetik veriyle öğrenme etkisidir; sıkıştırma mimarisinin tek başına daha doğru hale geldiği veya gerçek metinde kazanç sağladığı iddia edilmez.

## Hız ve eşdeğerlik kontrolü

Önceki teslimin kaynağı previous.c ve data.h anlık görüntülerinden derlenir. Eski motor + eski ağırlıklar, yeni motor + aynı eski ağırlıklar, yeni motor + yeni eğitim ağırlıkları sırayla çalıştırılır. -O3, bir ısınma ve dokuz ölçüm turu; her varyant iki süreçte dönüşümlü değerlendirilir. Son kaydedilen koşu başka eğitim, sanitizer veya değerlendirme işi olmadan tek başına çalıştırıldı.

| Varyant | Örnek başına medyan | Accuracy, seed 42 | NLL | Kullanılan tape düğümleri |
|---|---:|---:|---:|---:|
| Eski motor/ağırlık | 0.027485 ms | %69.875 | 1.027683 | 636384 B |
| Optimize motor, aynı ağırlık | 0.024877 ms | %69.875 | 1.027683 | 576480 B |
| Optimize motor, yeni eğitim | 0.025073 ms | %83.650 | 0.710864 | 576480 B |

Yeni ağırlıklarla süre %8.78 azaldı. Aynı eski ağırlıklarla kayıp ve tahmin sonuçları eşit kaldı. İnvariance, hız ve doğruluk kabul koşulları true.

Ölçüm sıkıştırmayı yeniden oluşturma ve autodiff grafiğini içerir. Tape boyutu yaklaşık %9,4 azalır; bu toplam RAM veya ayrılmış arena kapasitesi değildir. Cached önem skalerinin mantıksal temsil boyutu blok başına 8 bayt ekler. Bazı ara eğitim CSV'leri bu skaleri eski representation_bytes hesabına dahil etmez; güncel kaynak dahil eder. Parametre ve attention bloğu sayısı değişmez. Süreler sistem yüküne göre değişir; bütün uzunluk/donanımlar için garanti değildir.

## Diğer denemeler

Geliştirme verisinde seed=42, 100.000 örnek. Eğitimde değiştirilen seçenekler dosya adlarında ve CSV metadata'sında bulunur. Başarısız seçenekler varsayılan yapılmadı.

| Deney | Accuracy | NLL |
|---|---:|---:|
| mass-42 | %38.075 | 1.588793 |
| sigmoid-42 | %38.650 | 1.572046 |
| product-42 | %37.875 | 1.598001 |
| mass-product-42 | %37.550 | 1.588134 |
| sigmoid-product-42 | %37.725 | 1.659541 |
| tied-42 | %64.725 | 1.317783 |
| tied-mask-42 | %63.575 | 1.335685 |
| sigmoid-tied-42 | %39.000 | 1.596212 |
| batch4-42 | %38.700 | 1.454100 |
| batch4-lr-42 | %71.050 | 0.850585 |
| batch8-lr-42 | %38.850 | 1.453734 |
| lr6-42 | %38.275 | 1.586076 |
| distinct-42 | %83.025 | 0.783311 |
| distinct-early-42 | %81.300 | 0.800946 |

mass-pool, blokların toplam önemini taşıyan birleşimli bir istatistik denemesi; sigmoid-pool ise token ilişkilerini birbirinden bağımsız kapılarla toplama denemesi. Bunlar daha az iş yapsa da doğrulukları kabul edilmedi. Pure product ilişki kodlaması, tied/tied-mask okuma, büyük batch ve yüksek öğrenme oranı da denendi. Batch4 + yüksek öğrenme oranı seed=42'de iyi görünse de diğer tohumlarda kötüleşti; varsayılan batch=1 ve öğrenme oranı 0,003 korunur.

Üç tohumlu geliştirme ortalamaları:

| Eğitim | Accuracy ortalaması |
|---|---:|
| distinct | %76.475 |
| distinct-early | %75.667 |
| batch4-lr | %61.867 |
| cached-distinct | %76.658 |

cached-distinct koşuları aynı matematiğin cache ile eğitimini kontrol eder. Nihai rapor ve önerilen dosya, cache kapalı eğitimden gelen distinct-42 ağırlıklarıyla cache açık çıkarımı kullanır. Küçük ağırlık/sonuç farklılıkları nedeniyle bu koşular birleştirilip tek sayı olarak sunulmaz.

## Kullanım ve doğrulama

```sh
make check
./divide --only divide --distinct yes --score-cache no --steps 100000 --seed 42 --save retrained --csv retrained.csv
./divide --load retrained.divide.weights --score-cache yes --eval-task mixed --eval-seed 700003 --tests 4000
python3 experiments/round3/verify_speed.py
```

Varsayılan distinct=yes, score-cache=yes. recommended.divide.weights yenilendi; önceki dosya previous-recommended.weights olarak korunur. Optimizer durumları checkpointte saklanmaz. CLI'de mass/sigmoid birleştirme, alternatif ilişki kodlaması/çıktı okuması, batch ve veri düzenleri deneme seçenekleri olarak kalır. DIVIDE1–7 checkpointleri desteklenir.

Kontroller: on sıkıştırıcı × beş okuma × iki ilişki kodlaması × iki geçmiş mimarisi için sonlu fark gradyanları; cache açık/kapalı bütün parametre gradyanları ve kayıp eşitliği; mass istatistiği birleştirme tutarlılığı; veri/cevap tutarlılığı; distinct örneklerde değer tekilliği; eksik batch sonunun işlenmesi; eğitim/checkpoint/tahmin ve eski checkpoint akışı; AddressSanitizer ve UndefinedBehaviorSanitizer.

Bu hâlâ dört anahtar, on sayı ve x/y/z dolgulu 20 karakterlik bir görevdir. Gerçek dil modelleme, daha karmaşık ilişkiler veya GPU sistemleri için doğruluk/hız kazanımı kanıtı değildir.
