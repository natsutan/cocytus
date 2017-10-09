SUBDIRS := img

.PHONY: doc $(SUBDIRS)
.PHONY: clean
all:manual.docx install.docx

$(SUBDIRS):
	$(MAKE) -C $@ -f Makefile $(MAKECMDGOALS)

manual.docx:manual.md $(SUBDIRS)	
	pandoc --self-contained -s manual.md -o manual.docx

install.docx:manual.md $(SUBDIRS)
	pandoc --self-contained -s install.md -o install.docx


clean:$(SUBDIRS)
	@rm -f *.docx
