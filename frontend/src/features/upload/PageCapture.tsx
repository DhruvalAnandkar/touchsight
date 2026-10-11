import type { RefObject } from "react";

type PageCaptureProps = {
  inputId: string;
  inputRef: RefObject<HTMLInputElement | null>;
  fileName: string | null;
  onFile: (file: File) => void;
};

function takeFile(fileList: FileList | null, onFile: (file: File) => void) {
  const file = fileList?.[0];
  if (file) onFile(file);
}

export function PageCapture({ inputId, inputRef, fileName, onFile }: PageCaptureProps) {
  return (
    <div className="flex w-full min-w-0 flex-col gap-3">
      <label
        htmlFor={inputId}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          takeFile(event.dataTransfer.files, onFile);
        }}
        className="flex min-h-44 w-full min-w-0 cursor-pointer flex-col items-start justify-center gap-2 overflow-hidden rounded-2xl border-2 border-dashed border-gold/50 bg-desk-2 px-5 py-6 break-words hover:border-gold"
      >
        <span className="max-w-full font-bold break-words text-paper">Drop an image here, or browse</span>
        <span className="max-w-full break-words text-mist">PNG, JPG, or WEBP. Up to 8 MB.</span>
        {fileName ? <span className="max-w-full font-bold break-all text-gold">{fileName}</span> : null}
        <input
          ref={inputRef}
          id={inputId}
          type="file"
          accept="image/*"
          className="sr-only"
          onChange={(event) => {
            takeFile(event.target.files, onFile);
            event.target.value = "";
          }}
        />
      </label>
      <div className="relative w-full min-w-0">
        <button
          type="button"
          tabIndex={-1}
          aria-hidden="true"
          className="pointer-events-none flex min-h-14 w-full items-center justify-center rounded-2xl bg-gold px-4 py-4 text-center text-base font-bold break-words text-gold-deep"
        >
          Take a photo of a page
        </button>
        <input
          type="file"
          accept="image/*"
          capture="environment"
          aria-label="Take a photo of a page"
          className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
          onChange={(event) => {
            takeFile(event.target.files, onFile);
            event.target.value = "";
          }}
        />
      </div>
    </div>
  );
}
