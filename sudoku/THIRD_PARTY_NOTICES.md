# Third-party code

## Sudoku puzzle generator

Files: `puzzle_generator.py` and `advanced_sudoku_generator.py`.

Both files are unmodified copies of the files with the same names in
https://github.com/alicommit-malp/sudoku, at commit
`0617f5475803d151145d828b645428be1151d504` (committed 2024-09-14 by Ali Alp). In this
directory they are used only by `generate_data.py` and `smoke_test.py`, to sample
solved grids.

The upstream repository has no LICENSE file, and its source files carry no copyright
or licence header. Its README states: "This project is licensed under the MIT
License." The MIT permission notice below is based on that statement. Upstream gives
no copyright line, so the holder below is the author as shown on the upstream
repository (GitHub account `alicommit-malp`, name Ali Alp), and no year is given.

The two files are kept byte-identical to the code used for the paper, so they carry no
header of their own. The notice below applies to both of them.

```
MIT License

Copyright (c) Ali Alp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
