"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Upload, File, X, FileText, FileSpreadsheet, Database } from "lucide-react"
import { cn } from "@/lib/utils"

interface FileUploaderProps {
  onFileUpload: (file: File) => void
  acceptedFileTypes?: string
  maxSize?: number // in MB
}

export function FileUploader({
  onFileUpload,
  acceptedFileTypes = ".csv,.xlsx,.xls,.json",
  maxSize = 50,
}: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0])
    }
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0])
    }
  }

  const validateAndSetFile = (file: File) => {
    setError(null)

    // Check file type
    const fileExtension = file.name.split(".").pop()?.toLowerCase() || ""
    const isValidType = acceptedFileTypes.includes(`.${fileExtension}`)

    if (!isValidType) {
      setError(`Invalid file type. Accepted types: ${acceptedFileTypes}`)
      return
    }

    // Check file size
    const fileSizeInMB = file.size / (1024 * 1024)
    if (fileSizeInMB > maxSize) {
      setError(`File size exceeds the maximum limit of ${maxSize}MB`)
      return
    }

    setSelectedFile(file)
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    setIsUploading(true)

    try {
      // In a real implementation, you might want to upload the file to a server here
      // For now, we'll just simulate a delay and call the onFileUpload callback
      await new Promise((resolve) => setTimeout(resolve, 1000))
      onFileUpload(selectedFile)
    } catch (error) {
      setError("An error occurred during upload")
      console.error(error)
    } finally {
      setIsUploading(false)
    }
  }

  const handleRemoveFile = () => {
    setSelectedFile(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const getFileIcon = (fileName: string) => {
    const extension = fileName.split(".").pop()?.toLowerCase()

    switch (extension) {
      case "csv":
        return <FileText className="h-8 w-8 text-blue-500" />
      case "xlsx":
      case "xls":
        return <FileSpreadsheet className="h-8 w-8 text-green-500" />
      case "json":
        return <Database className="h-8 w-8 text-amber-500" />
      default:
        return <File className="h-8 w-8 text-muted-foreground" />
    }
  }

  return (
    <div className="space-y-4">
      <div
        className={cn(
          "border-2 border-dashed rounded-lg p-8 text-center transition-colors",
          isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/25",
          selectedFile ? "py-4" : "py-12",
        )}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {!selectedFile ? (
          <div className="flex flex-col items-center justify-center">
            <Upload className="h-10 w-10 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">Drag and drop your file here</h3>
            <p className="text-sm text-muted-foreground mb-4">or click to browse your files</p>
            <Button variant="outline" onClick={() => fileInputRef.current?.click()}>
              Select File
            </Button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileInputChange}
              accept={acceptedFileTypes}
              className="hidden"
            />
            <p className="text-xs text-muted-foreground mt-4">
              Accepted file types: {acceptedFileTypes} (Max size: {maxSize}MB)
            </p>
          </div>
        ) : (
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              {getFileIcon(selectedFile.name)}
              <div className="ml-4 text-left">
                <p className="font-medium">{selectedFile.name}</p>
                <p className="text-sm text-muted-foreground">{formatFileSize(selectedFile.size)}</p>
              </div>
            </div>
            <Button variant="ghost" size="icon" onClick={handleRemoveFile} disabled={isUploading}>
              <X className="h-5 w-5" />
            </Button>
          </div>
        )}
      </div>

      {error && <div className="bg-destructive/10 text-destructive px-4 py-2 rounded-md text-sm">{error}</div>}

      {selectedFile && !error && (
        <div className="flex justify-end">
          <Button onClick={handleUpload} disabled={isUploading}>
            {isUploading ? (
              <>
                <span className="animate-spin mr-2">⟳</span>
                Uploading...
              </>
            ) : (
              <>
                <Upload className="mr-2 h-4 w-4" />
                Upload File
              </>
            )}
          </Button>
        </div>
      )}
    </div>
  )
}

function formatFileSize(bytes: number): string {
  if (bytes === 0) return "0 Bytes"
  const k = 1024
  const sizes = ["Bytes", "KB", "MB", "GB"]
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
}
