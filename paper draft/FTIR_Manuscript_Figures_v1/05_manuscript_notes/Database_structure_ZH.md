# 数据库文件结构与处理关系

原始文件保持不变；“删除重复”指生成新的精确去重副本，不是删除原始文件。

```
dataset/
  FTIR_PLastics500_c4.csv                  原始3,000条
  FTIR_PLastics500_c8.csv                  原始3,000条
  deduplicated/
    FTIR_PLastics500_c4_deduplicated.csv   2,649条
    FTIR_PLastics500_c8_deduplicated.csv   2,390条
    FTIR_PLastics500_c4_row_mapping.csv   原始行→保留行/ID的映射
    FTIR_PLastics500_c8_row_mapping.csv
    summary.json                         原始计数、逐类去重数、源文件哈希
results/bidirectional_c4_c8_final/
  paired_group_representatives.csv        2,047组，各组C4/C8代表行索引与ID
  all_original_ID_groups.csv              全部原始ID→保守记录组
  common_grid.npy                        1,868个固定物理波数坐标
  c4_native_physical_coordinates.npy      3,736个原生坐标
  c8_native_physical_coordinates.npy      1,869个原生坐标
  <held-out-task>/seed_<0|1|2>/
    split.json                           base train/validation/query，新类support/query池
    episodes.json                        200次配对支持/查询索引，1/5/10/20为同一支持顺序的前缀
    sample_manifest.csv                  记录组及其角色
```

## CSV 的真实布局

光谱CSV第1行为列索引header，第2行为波数元数据；从第3行开始才是光谱记录。每条记录的第0列为source ID，第1列为聚合物名称，第2列为class ID，第3列起为光谱值；波数应读取第2行的第3列起，不能把列索引header当波数。C4有3,736个原生光谱值，C8有1,869个。

row_mapping.csv字段：source_csv_row, source_id, polymer, class_id, status, retained_source_csv_row, retained_source_id。

paired_group_representatives.csv中的c4_index/c8_index指向去重载入后的行数组，不是原始CSV行号；source ID与CSV行号也不是同一概念。禁止按去重后行号猜测跨分辨率同一实物。

## 当前计数

| 聚合物 | 原始每库 | C4精确去重 | C8精确去重 | 最终配对记录组 |
| --- | ---: | ---: | ---: | ---: |
| PET | 500 | 497 | 496 | 494 |
| HDPE | 500 | 497 | 497 | 495 |
| PVC | 500 | 400 | 300 | 201 |
| LDPE | 500 | 499 | 299 | 299 |
| PP | 500 | 497 | 499 | 497 |
| PS | 500 | 259 | 299 | 61 |

谱值单位尚未源文档确认；图示使用预处理数值单位。去重依据为精确数值相等；保守配对组依据共享原始ID、重复别名与最终共同网格精确指纹，无物理实物独立性保证。
