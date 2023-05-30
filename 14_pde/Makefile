.PHONY: diff
diff:
	make cpp py
	diff py cpp

.PHONY: py
py: 10_cavity.py
	python3 10_cavity.py > py

.PHONY: cpp
cpp: 10_cavity.cpp
	g++ 10_cavity.cpp
	./a.out > cpp
