# divide.c

C11 ile yazılmış küçük bir CPU eğitim ve karşılaştırma aracı. Yakın tokenler ayrıntılı kalır; uzak geçmiş 2, 4, 8, 16… tokenlik ikili bloklarda öğrenilmiş temsillere sıkıştırılır. Yeni sorgu bu bloklara attention uygular. Harici C kütüphanesi gerektirmez.

Bu sürüm 20 karakterlik bir anahtar–sayı hatırlama görevi üzerinde araştırma prototipidir. Genel amaçlı sohbet modeli veya tam GPT değildir.

## Kullanım

```sh
make
make check
# Varsayılanlar: 100.000 adım, 4.000 test, karma yerleşim, yeni yöntem.
./divide --only divide --save my-model --csv my-results.csv
# Aynı veri/başlangıç ve parametre sayısıyla normal attention da eğitilir:
./divide --only both --save comparison --csv comparison.csv
# Bu çalışma sırasında eğitilmiş, seed=42 modeli:
./divide --load recommended.divide.weights --predict 'a3;b7;c1;d9;xyzxyz?a='
./divide --load recommended.divide.weights --eval-task scattered --eval-seed 700001 --tests 4000
```

`--help` bütün seçenekleri gösterir. Kaydedilmiş normal attention modelini değerlendirme/tahmin için `--baseline` eklenir. `--only baseline` yalnızca normal modeli eğitir. `--only divide` yalnızca sıkıştırmalı modeli eğitir. `--only both` varsayılandır.

`--load` ile `--predict` verilmezse kaydedilmiş model yeni üretilen örneklerde değerlendirilir. Bu akışta `--context`, `--recent`, `--tests`, `--eval-task`, `--eval-seed` ve `--csv` kullanılabilir. Tahmin girdisinin sonu `?a=`, `?b=`, `?c=` veya `?d=` olmalıdır. Desteklenen karakterler `abcd0123456789;=?xyz`.

## Ana yöntem

Her tokenin öğrenilmiş bir yaprak vektörü vardır. Varsayılan yöntemde bu vektör önceki tokenle bileşen bazında etkileştirilir:

```
encoded_current = current + current * rotate(previous)
```

Yalnızca önceki token kullanılır; hedef karakter bu işlemde yer almaz. Böylece komşu anahtar/sayı ilişkisi, ikili blok sınırının neresine denk geldiğinden bağımsız olarak kodlanabilir. Token ve token çifti hesapları aynı ileri hesap içinde paylaşılır; optimizer güncellemeleri arasında cache tutulmaz.

Uzak geçmişin eşit uzunluktaki komşu blokları birleştirilir. İki çocuğun önem skorları öğrenilmiş bir vektörle hesaplanır; iki skorun softmax ağırlıklarıyla tek temsil üretilir. Bu birleştirme her seviyede uygulanır. Sorgu, az sayıdaki bloktan aldığı temsili bileşen bazında ağırlıklandırır: `context * query`.

Blok başlangıç konumu ve uzunluğu ayrıca saklanır ve öğrenilmiş özelliklerle key/value girdilerine eklenir. Örneğin 64 token ve 8 yakın token için 56 eski token 32+16+8 büyüklüğünde üç bloğa iner: toplam 11 attention bloğu. Alt parçaları yeniden açma veya tam attention'a geri dönme uygulanmaz.

12 boyut ve 1.112 öğrenilebilir parametre kullanılır. Gösterilen bilgi seçimi ayrı parametre eklemez; mevcut uzunluk özelliği vektörünü kullanır. Aynı seçeneklerle normal attention modeli de aynı yaprak/çift kodlayıcı ve çıktı okumasını kullanır; fark geçmişin birleştirilmesidir.

## Denenebilen yöntemler

| `--compressor` | Birleştirme |
|---|---|
| `legacy` | Simetrik ortalama + öğrenilmiş matris + tanh |
| `ordered` | Sağ vektörü döndürme + öğrenilmiş matris + tanh; önceki sürüm |
| `residual` | Öğrenilmiş dönüşümün yanında doğrudan taşıma yolu |
| `local` | İlk 8 tokene kadar dönüşüm; üst seviyelerde toplama |
| `binding` | Döndürmeli birleşime bileşen bazında çarpımlı etkileşim |
| `selective` | İlk 8 tokene kadar dönüşüm; üstte öğrenilmiş bilgi seçimi |
| `associative` | Komşu token ilişkisi kodlama + ikili toplama |
| `associative-selective` | Komşu ilişki kodlama + her seviyede öğrenilmiş bilgi seçimi; varsayılan |
| `mass-pool` | Blokta toplam önem kütlesini taşıyan birleştirme; deneysel |
| `sigmoid-pool` | Token ilişkilerini bağımsız kapılarla ağırlıklandırıp toplama; deneysel |

`--readout legacy|gated|product|tied|tied-mask` çıktı okumasını seçer; varsayılan `product`. `--init random|identity` başlangıç ağırlıklarını seçer; varsayılan `identity`, yaprak matrisini birim matris, bias ve konum/uzunluk vektörlerini sıfır başlatır. Bütün ağırlıklar yine öğrenilir. `--lr 0.003` başlangıç öğrenme oranıdır. `--schedule constant|cosine` varsayılan `cosine`; son öğrenme oranı başlangıcın %10'una iner. `--curriculum yes` ilk yarıda bağlam uzunluğunu 16'dan istenen uzunluğa artıran deneysel eğitimdir; varsayılan `no`.

Eski sürümün mimarisini ve eğitimini yeniden kurmak için:

```sh
./divide --only divide --compressor ordered --readout gated --init random \
  --schedule constant --distinct no --score-cache no --task fixed --steps 20000 --tests 4000 --seed 42
```

## Veri ve karşılaştırma

Her örnekte a/b/c/d anahtarlarının dört rastgele sayısı ve x/y/z dolgu karakterleri vardır. Son `?a=` benzeri sorgunun ardından gelecek sayı tahmin edilir. Eğitim yalnızca bu son konumdaki next-token kaybını kullanır.

- `--task fixed`: Eşleşmeler başlangıçta a,b,c,d sırasında.
- `--task shuffled`: Eşleşmelerin sırası değişir.
- `--task scattered`: Sıra ve eşleşmeler arasındaki dolgu miktarı değişir.
- `--task mixed`: Bu üç düzenin karışımı; varsayılan.

`--distinct yes|no|early` yalnız eğitim sayılarının dağılımını değiştirir. Yeni varsayılan `yes`, dört anahtarın sayılarının farklı olmasını sağlar. `early` yalnız ilk yarıda bu kısıtı uygular; `no` önceki bağımsız sayı seçimini korur. Değerlendirmede bu kısıt her zaman kapalıdır; tekrar eden sayılar da bulunur. Bu seçenek sentetik görevin eğitim düzenidir, modelin çıkarımında cevap veren bir kural değildir.

`--score-cache yes|no` önem puanlarının tekrar hesaplanmasını azaltır; varsayılan `yes`. İkili birleşimde ebeveyn puanı doğrusal yapıdan `sqrt(2) * (wa*score_left + wb*score_right)` ile taşınır. Aynı ağırlıklarda ileri sonuç ve tüm parametre gradyanları korunur. Blok başına bir skaler çalışma istatistiği taşınır; attention bloğu, vektör boyutu veya parametre eklenmez.

`--binding additive|product` ilişki kodlamasında tekil tokenin ayrıca taşınmasını seçer; varsayılan `additive`, denenmiş `product` daha iyi sonuç vermedi. `--batch N` 1..64 örneğin gradyanını ortalayıp optimizer güncellemesi yapar; varsayılan 1. `--steps` eğitim örneği sayısıdır: batch=4, steps=100000 için 25.000 optimizer güncellemesi yapılır. Son eksik grup da güncellenir.

`--eval-task` eğitim görevinden farklı bir değerlendirme düzeni seçebilir. `--seed` başlangıç ağırlıklarını, `--data-seed` eğitim verisini, `--eval-seed` değerlendirme verisini seçer. Varsayılanlar 42 / 12345 / 987654321. Eğitim/değerlendirme tohumları mimariler arasında aynı tutulur. Yöntem seçimi için kullanılmayan ek değerlendirmede 700001 tohumu kullanıldı.

En son karşılaştırmada, 100.000 eğitim örneği ve üç başlangıç tohumunun ayrı karma değerlendirme verisindeki ortalaması önceki önerilen modelde %69,2, yeni eğitimde %78,0 oldu. Eşleşmelerin konumları rastgele dağılmış düzende %71,8 → %81,3. Bunlar gerçek dil modelleme doğruluğu değildir. Önceki büyük mimari değişikliği [ikinci deney raporunda](experiments/round2/REPORT.md), son eğitim/hız iyileştirmeleri ve başarısız denemeler [üçüncü raporda](experiments/round3/REPORT.md).

## Ölçümler ve sınırlar

`train_seconds` veri üretimi, ileri/geri hesap ve optimizer dahil eğitim süresidir. `eval_ms_per_example` sıkıştırmayı yeniden oluşturma ve autodiff grafiğini kurma dahil örnek başına süredir. GPU, üretim KV cache veya yalnız attention süresi değildir.

`nll` tüm 20 karakter üzerinde ortalama negatif log olasılıktır; düşük daha iyidir. `accuracy` en yüksek olasılıklı karakterin hedefle eşleşmesidir. `attention_blocks` bakılan temsil sayısıdır. `representation_bytes` yalnız blok vektörleri, başlangıç/uzunluk alanları ve etkinse cached önem skalerinin hesaplanan boyutudur. Eski deney CSV'lerinin bir kısmı önem skalerini bu hesaba dahil etmez. `autodiff_peak_bytes` kullanılan en yüksek düğüm sayısı × düğüm boyutudur; ayrılmış arena kapasitesi, optimizer, token/çift cache, stack ve toplam RAM dahil değildir. Yeni motorun düğümleri 32 bayt, önceki motorunki 40 bayttır.

İkili blokları kurma maliyeti hâlâ lineerdir. Kalıcı streaming cache uygulanmaz. Toplam O(log n) veya kayıpsız sıkıştırma iddiası yoktur. Çift kodlayıcı yerel sıralamayı temsil eder; bütün metnin sırasını kayıpsız saklamaz.

Sabit konumlu görevde uzmanlaşmış `specialized.divide.weights` farklı bir yöntem ve eğitim kullanır; konumlar değiştiğinde kötüleşir. Önerilen model `recommended.divide.weights` karma yerleşimle eğitilmiştir. Önceki `model.*.weights` ve `improved.*.weights` korunmuştur. Checkpointler okunabilir metindir, ağırlıklar ve mimari seçimi doğrulanarak yüklenir; DIVIDE1–7 sürümleri desteklenir. Optimizer durumu kaydedilmez; eğitim kaldığı yerden sürdürülmez.

## Doğrulama ve hız koşulu

```sh
make check
cc -O1 -g -std=c11 -fsanitize=address,undefined test.c -lm -o test-sanitize
./test-sanitize
python3 experiments/round3/verify_speed.py
```

`make check`, on sıkıştırıcı, beş çıktı okuması ve iki ilişki kodlamasında iki mimarinin seçilmiş gradyanlarını sonlu farklarla denetler. Rastgele veri düzenlerinin anahtar/cevap tutarlılığı, blok sayıları, sıralama özellikleri ve checkpoint/tahmin akışı kontrol edilir.

Hız aracı Python 3 gerektirir. Önceki sürümün motoruyla yeni motoru aynı -O3 seçenekleriyle derler. Aynı bütçede eğitilmiş seed=42 modellerini 4.000 ayrı karma örnekte değerlendirir. Bir ısınma turundan sonra dokuz tur medyanını alır; eski/yeni süreçleri iki kez dönüşümlü çalıştırır. Yeni medyan süre eskiyi aşarsa veya doğruluk artmazsa başarısız çıkar. Sonuç `experiments/round3/speed-results.json` dosyasındadır. Bu ölçüm bu makine ve görev için geçerlidir; bütün donanım/uzunluklar için hız garantisi değildir.

Tüm yöntemlerin kısa deneylerini yeniden çalıştırmak için `python3 experiments/round2/run_experiments.py`; karma veri üzerinde üç tohumlu uzun karşılaştırma için `--full` eklenir. Bu komutlar mevcut kanıt dosyalarını değiştirmeden `experiments/round2/reproduced/` dizinine yazar.

Üçüncü deneydeki önerilen modeli yeniden eğitmek için:

```sh
./divide --only divide --distinct yes --score-cache no --steps 100000 --seed 42 --save retrained --csv retrained.csv
./divide --load retrained.divide.weights --score-cache yes --eval-task mixed --eval-seed 700003 --tests 4000
```

`recommended.divide.weights` bu ayarların eğitilmiş seed=42 modelidir. Skaler cache ile de üç tohumlu eğitim çalıştırıldı; sonuçlar rapordadır. Önceki önerilen ağırlıklar `experiments/round3/previous-recommended.weights` dosyasında korunur. Üçüncü turda doğruluk kazanımı eğitim verisinin düzeninden gelir; hız kazanımı aynı model hesabını paylaşmaktan gelir. Gerçek metinler üzerinde bir kazanım henüz ölçülmedi.

Üçüncü turda kısa deneyleri yeniden çalıştırma: `python3 experiments/round3/run_experiments.py`; üç tohum ve başarısız alternatifleri de çalıştırma: `--full`. Sonuçlar `experiments/round3/reproduced/` dizinine yazılır. `--score-cache` aynı model hesabını yeniden kullanır; kayan noktalı gradyan toplama sırasını değiştirdiği için uzun eğitimde ağırlıklar çok küçük farklılıklarla ilerleyebilir. Sabit ağırlıklarla tahmin/kayıp eşitliği ve parametre gradyanları ayrıca doğrulanır.
