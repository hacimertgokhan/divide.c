# İkinci yöntem deneyi

Ana ikili blok sıkıştırması korunarak farklı birleştirme ve eğitim yöntemleri denendi. Son seçilen yöntem komşu token ilişkilerini önce kodlar, ardından her ikili blok seviyesinde öğrenilmiş bilgi seçimi yapar. Yeni parametre, bellek slotu veya attention bloğu eklenmedi.

## Yalnız birleştirme değişikliği: 20.000 adım

Sabit yerleşim, başlangıç tohumu 42, aynı veri ve 4.000 geliştirme örneği. Okuma gated, başlangıç random, öğrenme oranı sabit.

| Yöntem | Doğruluk | NLL |
|---|---:|---:|
| ordered | %43.075 | 1.7293 |
| residual | %40.875 | 1.5273 |
| local | %51.225 | 1.7790 |
| binding | %39.525 | 1.6668 |
| selective | %69.550 | 0.9425 |

Sabit görevde selective ile 50.000 adım ve cosine eğitim, üç başlangıç tohumunda %90,650 / %99,225 / %88,825 doğruluk verdi (ortalama %92,900). Ancak seed 42 sabit görev modeli, sırası değişmiş örneklerde %20,325, konumları dağılmış örneklerde %10,850 doğruluğa düştü. Bu yüzden en yüksek sabit görev yüzdesini genel başarı olarak kabul etmedik. Model specialized.divide.weights olarak ayrı saklandı.

## Sıra ve yerleşim değişirken bilgi koruma

associative, token yaprağına önceki tokenin döndürülmüş vektörüyle çarpımlı etkileşim ekler. associative-selective buna her ikili seviyede öğrenilmiş iki çocuk önem ağırlığı ekler. Hiçbir token türü için sabit cevap, anahtar/sayı çözümleyicisi veya kurallı yanıt seçimi uygulanmaz. İşlem bütün token çiftlerine uygulanır; hedef yalnız kayıpta kullanılır.

| Karma veri deneyi, seed 42 | Eğitim adımı | Doğruluk |
|---|---:|---:|
| ordered-mixed-42 | 50000 | %22.425 |
| local-mixed-42 | 50000 | %28.975 |
| selective-mixed-42 | 50000 | %29.275 |
| associative-mixed-42 | 50000 | %37.550 |
| associative-selective-mixed-42 | 50000 | %53.425 |
| associative-selective-mixed-long-42 | 100000 | %57.250 |
| identity-mixed-42 | 100000 | %65.725 |
| product-mixed-42 | 100000 | %56.500 |
| identity-product-mixed-42 | 100000 | %70.575 |
| lr-mixed-42 | 100000 | %50.225 |

Diğer eğitim deneyleri de saklandı: 20.000 adımda cosine ordered %39,825, cosine local %41,900, bağlam uzunluğu müfredatıyla local %36,925 verdi. Kademeli eğitim bu koşuda yardımcı olmadı. Düşük başlangıç öğrenme oranı da seçilmedi.

identity başlangıçta yaprak matrisini birim matris, bias ve konum/uzunluk vektörlerini sıfır yapar; eğitimde bu ağırlıklar serbesttir. product, sorguyla etkileşmeyen context toplama yolunu kaldırır. Böylece son seçimin tamamı tek bir birleştirme değişikliğine atfedilmez: ilişkisel kodlama, seçmeli birleştirme, başlangıç ve çıktı okumasının birlikte kullanıldığı bir yöntemdir.

## Yöntem seçiminde kullanılmayan veri

Her iki taraf 100.000 adımda aynı karma eğitim dağılımıyla eğitildi. Üç başlangıç tohumu 42, 43 ve 44; eğitim veri tohumu 12345. Geliştirme veri tohumu 987654321 kullanıldı. Aşağıdaki 4.000 örneklik değerlendirme tohumu 700001, yöntem seçimi tamamlandıktan sonra kullanıldı. Eski yöntem ordered/gated/random, yeni yöntem associative-selective/product/identity; her ikisinde cosine ve aynı başlangıç öğrenme oranı 0,003. Parametre bütçesi 1.112, boyut 12 ve 64 token için blok sayısı 11.

| Ayrı değerlendirme düzeni | Eski, 3 tohum ortalaması | Yeni, 3 tohum ortalaması |
|---|---:|---:|
| fixed | %45.267 | %67.467 |
| shuffled | %29.250 | %66.725 |
| scattered | %16.325 | %71.792 |
| mixed | %30.417 | %68.883 |

Karma ayrı değerlendirmede tek tek tohumlar:

| Başlangıç tohumu | Eski | Yeni |
|---|---:|---:|
| 42 | %30.775 | %68.650 |
| 43 | %31.125 | %61.475 |
| 44 | %29.350 | %76.525 |

Bunlar başlangıç tohumlarına göre değişen gerçek küçük model sonuçlarıdır; yalnızca en iyi tohum seçilip sunulmadı. recommended.divide.weights standart seed=42 modelidir. Bu modeller yalnızca 64 tokenlik örneklerde eğitildi. Aynı seed=42 modelinin başka uzunluklarda ayrı karma değerlendirmesi:

| Bağlam uzunluğu | Doğruluk | Blok |
|---|---:|---:|
| 32 | %66.025 | 10 |
| 128 | %67.050 | 12 |
| 256 | %66.425 | 13 |

Uzunluk arttıkça NLL kötüleşiyor; doğruluk korunması tüm belirsizliğin veya bilgilerin iyi korunduğu anlamına gelmez. Bunlar hâlâ dört anahtar ve on sayıdan oluşan sentetik görevlerdir. Gerçek metin, kod veya genel next-token modellemesi için sonuç çıkarmıyoruz.

## Hız koşulu

Önceki teslim edilen motor previous.c anlık görüntüsünden derlendi. Karma veriyle aynı bütçede eğitilmiş eski modelin DIVIDE4 dosyası, ağırlıklar değişmeden DIVIDE3 başlığına dönüştürüldü; böylece eski motor aynı mimarinin ağırlıklarını okuyabildi. Yeni motor güncel kaynaktan derlendi. İkisi de -O3, aynı 4.000 ayrı karma örnek, ısınma + dokuz ölçüm turu, iki dönüşümlü süreç. Son rapor diğer eğitim/değerlendirme süreçleri tamamlandıktan sonra tek başına çalıştırılan kontrolün sonucudur.

| Ölçüm | Önceki motor/model | Yeni motor/model |
|---|---:|---:|
| Örnek başına süre | 0.037425 ms | 0.026493 ms |
| accuracy | 0.30775 | 0.6865 |
| nll | 2.184749909 | 1.070041286 |
| parameters | 1112 | 1112 |
| blocks | 11 | 11 |
| tape_bytes | 1311720 | 636384 |

Ölçülen süre %29.21 azaldı; hız ve doğruluk kabul koşulları geçti. Yeni autodiff motoru saklanan türevler yerine işlem türünü tutarak 40 baytlık düğümü 32 bayta indirdi. Doğrusal katmanlarda gereksiz ilk sıfır/toplama düğümleri kaldırıldı. Bunlar daha iyi bellek yönteminin yanında hızı korumaya yardımcı olan genel motor optimizasyonlarıdır; yalnızca algoritmik avantaj olarak sunulmaz.

Süreler sıkıştırmayı yeniden kurma ve autodiff grafiğini içerir. tape_bytes kullanılan düğümlerin boyutudur, toplam RAM değildir. Token çifti cache ve stack bu ölçüme dahil değildir. Hız tüm donanım, veri veya uzunluklar için garanti edilmez.

## Dosyalar ve yeniden çalıştırma

- divide.c: sekiz sıkıştırıcı, üç okuma, eğitim ve değerlendirme seçenekleri.
- data.h: ortak veri üreticisi; sabit, karışık ve dağılmış düzenler.
- recommended.divide.weights: yeni karma-veri seed=42 modeli.
- specialized.divide.weights: sabit görevde uzman model; yerleşim değişiminde başarısız.
- holdout-results.json: bütün ayrı değerlendirme sonuçları.
- speed-results.json: tekrarlı süre ve kabul koşulları.
- run_experiments.py: kısa yöntem taraması; --full üç tohumlu uzun karşılaştırma.
- verify_speed.py: eski/yeni motorla hız kabul kontrolü.

Gradyan kontrolleri sekiz sıkıştırıcı × üç okuma × iki geçmiş mimarisini kapsar. Rastgele veri üreticisinde cevap ve yerleşim tutarlılığı, blok sayısı ve sıralama özellikleri denetlendi. Eğitim/checkpoint/tahmin akışı, eski checkpoint yükleme ve AddressSanitizer/UndefinedBehaviorSanitizer kontrolleri geçti.
