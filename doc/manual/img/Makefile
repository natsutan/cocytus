all:cocytus_arch.png
.PHONY : clean


cocytus_arch.png: cocytus_arch.dot
	dot -T png cocytus_arch.dot -o cocytus_arch.png

clean:
	@rm -f *.png

