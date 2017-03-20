SUBDIRS := img

.PHONY: doc $(SUBDIRS)
all:manual.docx

$(SUBDIRS):
	$(MAKE) -C $@ -f Makefile

manual.docx:manual.md $(SUBDIRS)
	pandoc --self-contained -s manual.md -o manual.docx


