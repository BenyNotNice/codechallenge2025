 # PyDay 2025
 ### Code Challenge: Forensic STR Parent-Child Relationship Detector

 **Challenge**  
 Develop a highly scalable Python program that identifies likely single parent-child relationships (father or mother only) within a large mixed dataset of DNA profiles.

 The dataset consists of approximately 500,000 known STR profiles (parents and children mixed together, roles unknown) plus approximately 40 unknown query profiles. For each query profile, the program must search the large database and return a ranked list of the top candidate matches that could be either the parent of the query individual or the child of the query individual (bidirectional relationship detection).

 **Key Biological Concepts (Integrated)**  
 - **STR (Short Tandem Repeat)**: Highly variable regions in human DNA where a short sequence (typically 2–6 bases) is repeated a variable number of times. These are the primary markers used in forensic and paternity DNA testing.  
 - **Locus (plural: loci)**: A specific physical location on a chromosome containing an STR marker. Standard forensic panels use 13–24 loci (e.g., D8S1179, TH01, vWA, FGA, D21S11, etc.).  
 - **Allele**: The number of repeats observed at a locus (e.g., 13, 14, or 14.2 for microvariants). Each person has two alleles per locus – one inherited from each biological parent.  
 - **Homozygous**: Both alleles at a locus are identical (e.g., 13,13). **Heterozygous**: The two alleles differ (e.g., 13,14).  
 - **Genotype / STR Profile**: The complete set of allele pairs across all tested loci for an individual.  
 - **Mendelian Inheritance**: A child must receive exactly one allele at each locus from the biological mother and one from the biological father. In single-parent testing, at least one of the child’s alleles must match one of the candidate parent’s alleles at every locus (unless explained by mutation).  
 - **Mutation**: A rare change in repeat count during meiosis (typical rate 0.001–0.004 per locus per generation), usually differing by ±1 repeat. Mutations allow small discrepancies without excluding a true relationship.  
 - **Missing Data / Allele Dropout / Partial Profiles**: Common in degraded or low-quantity DNA samples; some loci may have no result ("-") or only one allele may amplify.  
 - **Likelihood Ratio (LR) / Paternity Index**: The statistical measure used to evaluate evidence. For each locus, LR = (probability of observed alleles if parent-child) / (probability of observed alleles if unrelated). The Combined Likelihood Ratio (CLR or CPI) is the product of individual locus LRs. Typical interpretation: CLR > 10,000 indicates strong support; > 100,000 is considered virtually proof.  
 - **Allele Frequencies**: Population-specific frequencies required to calculate the “unrelated” probability. Rare alleles shared between individuals dramatically increase the LR.

 **Input Format**  
 CSV file with columns:  
 - `PersonID` (unique identifier)  
 - One column per locus (e.g., `D8S1179`, `TH01`, `vWA`, etc.)  
 Allele values formatted as:  
 - `"13,14"` (heterozygous)  
 - `"13"` or `"13,13"` (homozygous)  
 - `"-"` or blank (missing locus/data)

 **Requirements**  
 - Compute accurate per-locus likelihood ratios handling exact matches, ±1 step mutations, allele dropout, and complete mismatches.  
 - Calculate the overall Combined Likelihood Ratio (CLR) for each candidate pair.  
 - Support bidirectional matching (same logic works for “query is child” or “query is parent”).  
 - Handle partial profiles and missing data robustly.  
 - Scale to ~500,000 database profiles × ~40 queries (millions of comparisons) – naive pairwise comparison is too slow. Implement efficient candidate pre-filtering/indexing (e.g., using shared rare alleles, multi-locus signatures, or hashing) to achieve practical runtime.  
 - For each query, output the top 10 candidates ranked by CLR (descending), including: PersonID, CLR value, estimated posterior probability (using a reasonable prior, e.g., 50%), number of consistent loci, number of loci requiring mutation, and number of missing/inconclusive loci.  

 **Bonus features:**  
 - Support microvariants (e.g., 9.3)  
 - Tri-allelic loci, null alleles  
 - Visualize matches or export detailed reports

 This challenge mirrors real-world forensic bioinformatics and paternity testing systems used in laboratories and courts worldwide. Good luck uncovering the hidden family connections! 🧬

 ---

 ### How to Participate

 1. **Fork** this repository  
 2. Implement your solution in  
    `src/codechallenge2025/participant_solution.py`  
    → You only need to fill in the `match_single(query_profile, database_df)` function.  
    → The rest (file loading, looping, output formatting) is already provided for you.  
 3. Open a **Pull Request** to the `main` branch  
 4. GitHub Actions will automatically:  
    - Generate a fresh dataset  
    - Run your code  
    - Evaluate accuracy  
    - Update the live **Leaderboard.md** with your score  
    - Post detailed results in a comment on your PR  

 You can test locally with:
 ```bash
 make all
 ```
 This will generate the dataset, run your code, and update a local leaderboard.

 **Live Leaderboard**: [Leaderboard.md](./Leaderboard.md)
 ---
 If you are using AI agents or any LLMs in your solution,
 I would like to know your steps, planning, prompts, and any other details
 those tools provide, as well as what tools you are using, which models,
 and which PR belongs to you.
 Please email me at: a.tavallaie@gmail.com
 with subject: pyday2025

 ---

 ### چالش برنامه‌نویسی: تشخیص‌دهنده رابطه والد و فرزند تک‌والدی با استفاده از STR در پزشکی قانونی

 یک برنامه پایتون بسیار سریع و مقیاس‌پذیر بنویسید که بتواند روابط احتمالی والد-فرزندی (فقط یکی از والدین: پدر یا مادر) را در میان یک مجموعه داده بزرگ و مخلوط از پروفایل‌های DNA پیدا کند.

 **داده‌ها:**
 - حدود ۵۰۰٬۰۰۰ پروفایل DNA شناخته‌شده (والدین و فرزندان با هم مخلوط شده‌اند و مشخص نیست کدام والد و کدام فرزند است).
 - به علاوه حدود ۴۰ پروفایل DNA ناشناخته (پرس‌وجوها).

 برنامه باید برای هر یک از این ۴۰ پروفایل ناشناخته، کل پایگاه داده ۵۰۰٬۰۰۰ تایی را جستجو کند و لیستی از بهترین کاندیداهای احتمالی را به ترتیب امتیاز برگرداند. این کاندیداها می‌توانند **والد** فرد ناشناخته باشند یا **فرزند** او (جستجو در هر دو جهت انجام می‌شود).

 **مفاهیم مهم بیولوژیکی (به زبان ساده):**

 - **STR (تکرارهای کوتاه پشت سر هم):** بخش‌هایی از DNA که یک الگوی کوتاه (۲ تا ۶ حرفی) چندین بار تکرار می‌شود. تعداد تکرارها در افراد مختلف متفاوت است و از این تفاوت برای شناسایی استفاده می‌شود.
 - **لوکوس:** هر جایگاه خاص روی DNA که یک STR در آن قرار دارد. معمولاً در تست‌های پزشکی قانونی از ۱۳ تا ۲۴ لوکوس استفاده می‌شود (مثل D8S1179، TH01، vWA و غیره).
 - **آلل:** تعداد تکرارها در یک لوکوس. هر فرد دو آلل دارد (یکی از مادر و یکی از پدر به ارث برده).
 - **هوموزیگوت:** هر دو آلل یکسان هستند (مثل ۱۳ و ۱۳).  
   **هتروزیگوت:** دو آلل متفاوت هستند (مثل ۱۳ و ۱۴).
 - **پروفایل STR:** مجموعه تمام آلل‌های یک فرد در همه لوکوس‌ها.
 - **قانون وراثت:** فرزند دقیقاً یکی از آلل‌های هر لوکوس را از مادر و یکی از پدر می‌گیرد. بنابراین در رابطه والد-فرزندی واقعی، در هر لوکوس حداقل یکی از آلل‌های فرزند با یکی از آلل‌های والد مطابقت دارد (مگر در موارد نادر جهش).
 - **جهش:** تغییر بسیار نادر در تعداد تکرارها (معمولاً فقط ±۱) که هنگام انتقال به فرزند اتفاق می‌افتد.
 - **داده‌های ناقص:** در نمونه‌های قدیمی یا کم‌کیفیت DNA ممکن است برخی لوکوس‌ها نتیجه نداشته باشند یا فقط یک آلل دیده شود.
 - **نسبت احتمال (LR):** شاخصی آماری که نشان می‌دهد چقدر احتمال دارد دو نفر واقعاً والد و فرزند باشند در مقایسه با اینکه کاملاً بی‌ربط باشند. این نسبت برای هر لوکوس محاسبه شده و سپس همه با هم ضرب می‌شوند تا CLR (نسبت احتمال ترکیبی) به دست آید.  
   معمولاً CLR بالای ۱۰٬۰۰۰ یعنی احتمال بسیار بالا، و بالای ۱۰۰٬۰۰۰ تقریباً اثبات‌کننده رابطه است.
 - **فرکانس آلل‌ها:** آلل‌های نادر اگر بین دو نفر مشترک باشند، امتیاز را خیلی بالا می‌برند.

 **فرمت فایل ورودی (CSV):**
 - ستون اول: PersonID (شناسه منحصربه‌فرد هر فرد)
 - ستون‌های بعدی: نام هر لوکوس (مثل D8S1179، TH01 و ...)
 - مقدار هر سلول:
   - "13,14" → دو آلل متفاوت
   - "13" یا "13,13" → هر دو آلل یکسان
   - "-" یا خالی → داده موجود نیست

 **الزامات اصلی برنامه:**

 1. محاسبه دقیق نسبت احتمال (LR) برای هر لوکوس با در نظر گرفتن تطابق کامل، جهش ±۱، از دست رفتن آلل و عدم تطابق.
 2. محاسبه CLR کلی برای هر جفت.
 3. جستجو در هر دو جهت (پرس‌وجو می‌تواند والد یا فرزند باشد).
 4. مدیریت درست پروفایل‌های ناقص.
 5. سرعت بالا: باید از روش‌های هوشمند پیش‌فیلتر کردن استفاده کنید تا برنامه در زمان معقول اجرا شود.
 6. خروجی: ۱۰ کاندیدای برتر برای هر پرس‌وجو به همراه CLR، احتمال پسین، تعداد لوکوس‌های سازگار، جهش‌دار و نامشخص.

 **امکانات اضافی (امتیازی):**
 - پشتیبانی از میکروواریانت‌ها (مثل ۹٫۳)
 - لوکوس‌های سه‌آللی یا آلل‌های صفر
 - نمایش گرافیکی یا گزارش کامل

 این چالش بسیار شبیه به سیستم‌های واقعی مورد استفاده در آزمایشگاه‌های پزشکی قانونی و دادگاه‌ها برای پیدا کردن خویشاوندان یا تعیین نسب است.

 موفق باشید در کشف روابط خانوادگی! 🧬

 ---

 ### نحوه شرکت در چالش

 1. این مخزن را **Fork** کنید  
 2. راه‌حل خود را فقط در تابع  
    `match_single(query_profile, database_df)`  
    در فایل `src/codechallenge2025/participant_solution.py` پیاده‌سازی کنید  
    → بقیه کارها (خواندن فایل‌ها، حلقه روی پرس‌وجوها و فرمت خروجی) برای شما فراهم شده است  
 3. یک **Pull Request** به شاخه `main` باز کنید  
 4. GitHub Actions به‌طور خودکار:  
    - داده جدید تولید می‌کند  
    - کد شما را اجرا می‌کند  
    - دقت را ارزیابی می‌کند  
    - جدول امتیازات زنده (**Leaderboard.md**) را به‌روز می‌کند  
    - نتایج دقیق را در کامنت PR شما نمایش می‌دهد  

 برای تست محلی:
 ```bash
 make all
 ```

 **جدول امتیازات زنده**: [Leaderboard.md](./Leaderboard.md)

 ---

 اگر در راه‌حل خود از عامل‌های هوش مصنوعی (AI agents) یا مدل‌های زبانی بزرگ (LLMs) استفاده می‌کنید،
 دوست دارم مراحل کار، برنامه‌ریزی، پرامپت‌ها و هر جزئیات دیگری که این ابزارها ارائه می‌دهند را بدانم.
 همچنین از چه ابزارهایی استفاده کرده‌اید، کدام مدل‌ها را به کار برده‌اید
 و کدامیک از Pull Requestها متعلق به شماست.
 لطفاً به ایمیل a.tavallaie@gmail.com
 با موضوع pyday2025 ایمیل بزنید.