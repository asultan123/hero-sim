import subprocess
import regex as rx

ifmap_h = 10
ifmap_w = 10
k = 1
c_in = 3
f_out = 16
filter_count = 7
channel_count = 9

args = (
    "build/tests/estimation_enviornment",
    "--ifmap_h",
    f"{ifmap_h}",
    "--ifmap_w",
    f"{ifmap_w}",
    "--k",
    f"{k}",
    "--c_in",
    f"{c_in}",
    "--f_out",
    f"{f_out}",
    "--filter_count",
    f"{filter_count}",
    "--channel_count",
    f"{channel_count}",
)
# Or just:
# args = "bin/bar -c somefile.xml -d text.txt -r aString -f anotherString".split()
popen = subprocess.Popen(args, stdout=subprocess.PIPE)
popen.wait()
output = popen.stdout.read().decode()

print(output)
