text = input("متن را وارد کنید: ")

latin = arabic = persian = cyrillic = other = 0

for ch in text:
    code = ord(ch)

    if (65 <= code <= 90) or (97 <= code <= 122):
        latin += 1

    elif 0x0600 <= code <= 0x06FF:
        persian += 1

    elif 0x0400 <= code <= 0x04FF:
        cyrillic += 1

    else:
        other += 1

print("\nنتیجه:")
if latin > 0: print("حروف لاتین وجود دارد.")
if persian > 0: print("حروف فارسی/عربی وجود دارد.")
if cyrillic > 0: print("حروف سیریلیک وجود دارد.")
if other > 0: print("نماد/زبان‌های دیگر هم دیده شد.")
