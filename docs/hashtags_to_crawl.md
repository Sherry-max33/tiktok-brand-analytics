# 需要爬取的 Hashtag 全量列表

与 `configs/hashtags.yaml` 中 product_line_map / brand_style_map / product_category_map 对齐，保证每个映射 key 都有对应爬取来源。  
当前测试仅保留 4 个；正式扩大时用下面全量。

---

## Nike（16 个）

| # | hashtag | 说明 |
|---|---------|------|
| 1 | nike | 品牌主词 |
| 2 | jordan | 产品线/系列 |
| 3 | airmax | 产品线 |
| 4 | airforce1 | 产品线 |
| 5 | af1 | 产品线简称 |
| 6 | nikedunk | 产品线 |
| 7 | niketech | 产品线/风格 |
| 8 | nikeshoes | 品类 |
| 9 | nikesneakers | 品类 |
| 10 | nikerunning | 风格 |
| 11 | niketraining | 风格 |
| 12 | nikebasketball | 风格 |
| 13 | nikeoutfit | 风格/品类 |
| 14 | nikestyle | 风格 |
| 15 | nikefit | 品类 |
| 16 | nikelook | 品类 |

---

## Adidas（15 个）

| # | hashtag | 说明 |
|---|---------|------|
| 1 | adidas | 品牌主词 |
| 2 | adidassamba | 产品线/风格 |
| 3 | adidasgazelle | 产品线/风格 |
| 4 | adidasspezial | 产品线 |
| 5 | ultraboost | 产品线 |
| 6 | adidasforum | 产品线 |
| 7 | adidasoriginals | 产品线/风格 |
| 8 | tracksuit | 产品线/品类 |
| 9 | adidasshoes | 品类 |
| 10 | adidasfootball | 风格 |
| 11 | adidasrunning | 风格 |
| 12 | adidastraining | 风格 |
| 13 | adidasstyle | 风格/品类 |
| 14 | adidasoutfit | 品类 |
| 15 | adidaslook | 品类 |

---

## 合计

- **Nike**: 16 个  
- **Adidas**: 15 个  
- **合计**: **31 个** hashtag（与 project.yaml 中 keywords_target: 30 接近，可按需微调）

---

## 纯列表（复制进 YAML 或脚本用）

**Nike**

```
nike
jordan
airmax
airforce1
af1
nikedunk
niketech
nikeshoes
nikesneakers
nikerunning
niketraining
nikebasketball
nikeoutfit
nikestyle
nikefit
nikelook
```

**Adidas**

```
adidas
adidassamba
adidasgazelle
adidasspezial
ultraboost
adidasforum
adidasoriginals
tracksuit
adidasshoes
adidasfootball
adidasrunning
adidastraining
adidasstyle
adidasoutfit
adidaslook
```

---

说明：`normalize_tags` 中 adidassambas→adidassamba、adidasgazelles→adidasspezials 等为复数归一化，爬虫只需爬主形（如 adidassamba），复数形式会在文案中出现时被归一化到主形。
