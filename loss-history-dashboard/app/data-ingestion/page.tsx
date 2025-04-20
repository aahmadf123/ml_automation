"use client"

import { DashboardHeader } from "@/components/dashboard-header"
import { PageHeader } from "@/components/page-header"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { FileUploader } from "@/components/file-uploader"
import { DataPreview } from "@/components/data-preview"
import { ColumnMapper } from "@/components/column-mapper"
import { ValidationRules } from "@/components/validation-rules"
import { IngestionHistory } from "@/components/ingestion-history"

export default function DataIngestionPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <DashboardHeader />
      <div className="flex-1 p-6 space-y-6">
        <PageHeader
          title="Data Ingestion"
          description="Upload, validate, and process data for your machine learning models"
        />

        <Tabs defaultValue="upload" className="space-y-4">
          <TabsList>
            <TabsTrigger value="upload">Upload</TabsTrigger>
            <TabsTrigger value="preview">Preview</TabsTrigger>
            <TabsTrigger value="mapping">Mapping</TabsTrigger>
            <TabsTrigger value="validation">Validation</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-4">
            <FileUploader />
          </TabsContent>

          <TabsContent value="preview" className="space-y-4">
            <DataPreview />
          </TabsContent>

          <TabsContent value="mapping" className="space-y-4">
            <ColumnMapper />
          </TabsContent>

          <TabsContent value="validation" className="space-y-4">
            <ValidationRules />
          </TabsContent>

          <TabsContent value="history" className="space-y-4">
            <IngestionHistory />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
