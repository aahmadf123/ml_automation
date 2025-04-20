"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Search,
  Filter,
  Plus,
  Copy,
  Star,
  StarOff,
  Trash2,
  Edit,
  Save,
  X,
  Zap,
  FileCode,
  Database,
  BarChart,
  AlertTriangle,
  Clock,
} from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"

// Define the template interface
export interface FixTemplate {
  id: string
  name: string
  description: string
  category: "schema" | "data_quality" | "drift" | "performance" | "pipeline" | "custom"
  code: string
  tags: string[]
  autoFixCompatible: boolean
  confidenceThreshold?: number
  createdAt: Date
  updatedAt: Date
  usageCount: number
  isFavorite: boolean
  author: string
}

// Define the props for the component
interface FixTemplatesLibraryProps {
  onApplyTemplate?: (template: FixTemplate) => void
  onClose?: () => void
}

export function FixTemplatesLibrary({ onApplyTemplate, onClose }: FixTemplatesLibraryProps) {
  // State for templates
  const [templates, setTemplates] = useState<FixTemplate[]>([])
  const [filteredTemplates, setFilteredTemplates] = useState<FixTemplate[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [activeCategory, setActiveCategory] = useState<string>("all")
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false)
  const [showAutoFixOnly, setShowAutoFixOnly] = useState(false)
  const [selectedTemplate, setSelectedTemplate] = useState<FixTemplate | null>(null)
  const [isEditMode, setIsEditMode] = useState(false)
  const [editedTemplate, setEditedTemplate] = useState<Partial<FixTemplate>>({})
  const [isCreateMode, setIsCreateMode] = useState(false)

  // Load templates (in a real app, this would come from an API)
  useEffect(() => {
    // Mock data for templates
    const mockTemplates: FixTemplate[] = [
      {
        id: "template-1",
        name: "Fix Schema Mismatch in Property Value",
        description: "Converts property_value column from INTEGER to DECIMAL to handle decimal values correctly.",
        category: "schema",
        code: "ALTER TABLE properties\nALTER COLUMN property_value TYPE DECIMAL(12,2);",
        tags: ["schema", "type-conversion", "property-value"],
        autoFixCompatible: true,
        confidenceThreshold: 80,
        createdAt: new Date(2023, 5, 15),
        updatedAt: new Date(2023, 6, 2),
        usageCount: 42,
        isFavorite: true,
        author: "System",
      },
      {
        id: "template-2",
        name: "Handle Missing Claim Dates",
        description: "Fills missing claim_date values with the policy_start_date plus 30 days as a reasonable default.",
        category: "data_quality",
        code: "UPDATE claims\nSET claim_date = policy_start_date + INTERVAL '30 days'\nWHERE claim_date IS NULL AND policy_start_date IS NOT NULL;",
        tags: ["missing-values", "dates", "claims"],
        autoFixCompatible: true,
        confidenceThreshold: 75,
        createdAt: new Date(2023, 4, 10),
        updatedAt: new Date(2023, 4, 10),
        usageCount: 28,
        isFavorite: false,
        author: "System",
      },
      {
        id: "template-3",
        name: "Adjust Drift Threshold for Location Code",
        description: "Updates the drift detection threshold for location_code feature to reduce false positives.",
        category: "drift",
        code: "// Update drift threshold configuration\nconfig.setFeatureDriftThreshold('location_code', 0.08);\n\n// Recalibrate the drift detection model\ndriftDetector.recalibrate('location_code');",
        tags: ["drift", "threshold", "location"],
        autoFixCompatible: true,
        confidenceThreshold: 70,
        createdAt: new Date(2023, 3, 22),
        updatedAt: new Date(2023, 7, 5),
        usageCount: 15,
        isFavorite: true,
        author: "System",
      },
      {
        id: "template-4",
        name: "Fix Data Type Inconsistency in Zip Codes",
        description: "Standardizes zip code format by converting numeric values to 5-digit strings with leading zeros.",
        category: "data_quality",
        code: "UPDATE properties\nSET zip_code = LPAD(zip_code::text, 5, '0')\nWHERE zip_code ~ '^[0-9]+$';",
        tags: ["data-type", "zip-code", "standardization"],
        autoFixCompatible: true,
        confidenceThreshold: 85,
        createdAt: new Date(2023, 2, 18),
        updatedAt: new Date(2023, 2, 18),
        usageCount: 36,
        isFavorite: false,
        author: "System",
      },
      {
        id: "template-5",
        name: "Optimize Model Performance for High-Value Claims",
        description: "Adjusts model weights to improve prediction accuracy for high-value claims.",
        category: "performance",
        code: "// Adjust model weights for high-value claims\nmodel.adjustFeatureWeight('claim_amount', 1.5);\nmodel.adjustFeatureWeight('property_value', 1.3);\nmodel.adjustFeatureWeight('risk_factor', 1.2);\n\n// Retrain the model with adjusted weights\nmodel.retrain();",
        tags: ["model", "weights", "high-value", "claims"],
        autoFixCompatible: false,
        createdAt: new Date(2023, 6, 30),
        updatedAt: new Date(2023, 6, 30),
        usageCount: 7,
        isFavorite: false,
        author: "System",
      },
      {
        id: "template-6",
        name: "Fix Pipeline Timeout Issues",
        description:
          "Resolves timeout issues in the data processing pipeline by optimizing batch size and adding checkpoints.",
        category: "pipeline",
        code: "// Optimize batch size for data processing\nconfig.setBatchSize(500);\n\n// Add checkpoints to allow resume on failure\npipeline.enableCheckpoints(true);\npipeline.setCheckpointInterval(5000);\n\n// Increase timeout threshold\npipeline.setTimeout(300000); // 5 minutes",
        tags: ["pipeline", "timeout", "optimization"],
        autoFixCompatible: false,
        createdAt: new Date(2023, 5, 5),
        updatedAt: new Date(2023, 5, 5),
        usageCount: 12,
        isFavorite: false,
        author: "System",
      },
      {
        id: "template-7",
        name: "Standardize Date Formats",
        description: "Converts all date formats to ISO standard (YYYY-MM-DD) for consistency.",
        category: "data_quality",
        code: "// For PostgreSQL\nUPDATE claims\nSET claim_date = TO_CHAR(TO_DATE(claim_date, 'MM/DD/YYYY'), 'YYYY-MM-DD')\nWHERE claim_date LIKE '__/__/____';\n\nUPDATE claims\nSET claim_date = TO_CHAR(TO_DATE(claim_date, 'DD-Mon-YYYY'), 'YYYY-MM-DD')\nWHERE claim_date LIKE '__-___-____';",
        tags: ["dates", "standardization", "format"],
        autoFixCompatible: true,
        confidenceThreshold: 90,
        createdAt: new Date(2023, 4, 12),
        updatedAt: new Date(2023, 4, 12),
        usageCount: 31,
        isFavorite: true,
        author: "System",
      },
      {
        id: "template-8",
        name: "Remove Outliers from Training Data",
        description: "Identifies and removes statistical outliers from the training dataset to improve model accuracy.",
        category: "data_quality",
        code: "// Define outlier detection function using IQR method\nfunction removeOutliers(data, column) {\n  const values = data.map(row => row[column]).sort((a, b) => a - b);\n  const q1 = values[Math.floor(values.length * 0.25)];\n  const q3 = values[Math.floor(values.length * 0.75)];\n  const iqr = q3 - q1;\n  const lowerBound = q1 - 1.5 * iqr;\n  const upperBound = q3 + 1.5 * iqr;\n  \n  return data.filter(row => {\n    const value = row[column];\n    return value >= lowerBound && value <= upperBound;\n  });\n}\n\n// Apply to key numeric columns\ntrainingData = removeOutliers(trainingData, 'claim_amount');\ntrainingData = removeOutliers(trainingData, 'property_value');\ntrainingData = removeOutliers(trainingData, 'risk_score');",
        tags: ["outliers", "data-cleaning", "training-data"],
        autoFixCompatible: false,
        createdAt: new Date(2023, 3, 28),
        updatedAt: new Date(2023, 3, 28),
        usageCount: 19,
        isFavorite: false,
        author: "System",
      },
      {
        id: "template-9",
        name: "Fix Null Values in Risk Factors",
        description: "Replaces null risk factor values with calculated estimates based on property characteristics.",
        category: "data_quality",
        code: "// Calculate estimated risk factors based on property characteristics\nUPDATE properties\nSET risk_factor = (\n  CASE\n    WHEN property_type = 'residential' THEN 2.5\n    WHEN property_type = 'commercial' THEN 3.8\n    WHEN property_type = 'industrial' THEN 4.2\n    ELSE 3.0\n  END\n) * (\n  CASE\n    WHEN year_built < 1950 THEN 1.5\n    WHEN year_built BETWEEN 1950 AND 1980 THEN 1.2\n    WHEN year_built BETWEEN 1981 AND 2000 THEN 1.0\n    ELSE 0.8\n  END\n)\nWHERE risk_factor IS NULL;",
        tags: ["null-values", "risk-factor", "estimation"],
        autoFixCompatible: true,
        confidenceThreshold: 65,
        createdAt: new Date(2023, 5, 20),
        updatedAt: new Date(2023, 5, 20),
        usageCount: 24,
        isFavorite: false,
        author: "System",
      },
      {
        id: "template-10",
        name: "Optimize Feature Selection",
        description:
          "Updates the feature selection process to focus on the most predictive variables based on recent model performance.",
        category: "performance",
        code: "// Update feature importance thresholds\nconst featureImportance = await model.getFeatureImportance();\nconst topFeatures = featureImportance\n  .filter(f => f.importance > 0.05)\n  .map(f => f.name);\n\n// Update feature selection configuration\nconfig.setSelectedFeatures(topFeatures);\n\n// Retrain model with optimized feature set\nawait model.retrain({\n  features: topFeatures,\n  hyperparameters: {\n    learningRate: 0.01,\n    maxDepth: 6,\n    numEstimators: 100\n  }\n});",
        tags: ["feature-selection", "optimization", "model-performance"],
        autoFixCompatible: false,
        createdAt: new Date(2023, 7, 8),
        updatedAt: new Date(2023, 7, 8),
        usageCount: 5,
        isFavorite: true,
        author: "System",
      },
    ]

    setTemplates(mockTemplates)
    setFilteredTemplates(mockTemplates)
  }, [])

  // Filter templates based on search, category, and favorites
  useEffect(() => {
    let filtered = [...templates]

    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(
        (template) =>
          template.name.toLowerCase().includes(query) ||
          template.description.toLowerCase().includes(query) ||
          template.tags.some((tag) => tag.toLowerCase().includes(query)),
      )
    }

    // Filter by category
    if (activeCategory !== "all") {
      filtered = filtered.filter((template) => template.category === activeCategory)
    }

    // Filter by favorites
    if (showFavoritesOnly) {
      filtered = filtered.filter((template) => template.isFavorite)
    }

    // Filter by auto-fix compatibility
    if (showAutoFixOnly) {
      filtered = filtered.filter((template) => template.autoFixCompatible)
    }

    setFilteredTemplates(filtered)
  }, [templates, searchQuery, activeCategory, showFavoritesOnly, showAutoFixOnly])

  // Handle template selection
  const handleSelectTemplate = (template: FixTemplate) => {
    setSelectedTemplate(template)
    setIsEditMode(false)
    setEditedTemplate({})
  }

  // Handle applying a template
  const handleApplyTemplate = () => {
    if (selectedTemplate && onApplyTemplate) {
      // Increment usage count
      setTemplates((prev) =>
        prev.map((t) => (t.id === selectedTemplate.id ? { ...t, usageCount: t.usageCount + 1 } : t)),
      )
      onApplyTemplate(selectedTemplate)
    }
  }

  // Handle toggling favorite status
  const handleToggleFavorite = (templateId: string) => {
    setTemplates((prev) => prev.map((t) => (t.id === templateId ? { ...t, isFavorite: !t.isFavorite } : t)))
  }

  // Handle editing a template
  const handleEditTemplate = () => {
    if (selectedTemplate) {
      setEditedTemplate({ ...selectedTemplate })
      setIsEditMode(true)
    }
  }

  // Handle saving edited template
  const handleSaveTemplate = () => {
    if (isEditMode && selectedTemplate && editedTemplate) {
      const updatedTemplate = {
        ...selectedTemplate,
        ...editedTemplate,
        updatedAt: new Date(),
      }

      setTemplates((prev) => prev.map((t) => (t.id === selectedTemplate.id ? updatedTemplate : t)))
      setSelectedTemplate(updatedTemplate)
      setIsEditMode(false)
      setEditedTemplate({})
    }
  }

  // Handle creating a new template
  const handleCreateTemplate = () => {
    setSelectedTemplate(null)
    setIsEditMode(false)
    setIsCreateMode(true)
    setEditedTemplate({
      name: "",
      description: "",
      category: "custom",
      code: "",
      tags: [],
      autoFixCompatible: false,
      confidenceThreshold: 80,
      author: "User",
    })
  }

  // Handle saving a new template
  const handleSaveNewTemplate = () => {
    if (editedTemplate.name && editedTemplate.code) {
      const newTemplate: FixTemplate = {
        id: `template-${Date.now()}`,
        name: editedTemplate.name || "Untitled Template",
        description: editedTemplate.description || "",
        category: (editedTemplate.category as FixTemplate["category"]) || "custom",
        code: editedTemplate.code || "",
        tags: editedTemplate.tags || [],
        autoFixCompatible: editedTemplate.autoFixCompatible || false,
        confidenceThreshold: editedTemplate.confidenceThreshold,
        createdAt: new Date(),
        updatedAt: new Date(),
        usageCount: 0,
        isFavorite: false,
        author: editedTemplate.author || "User",
      }

      setTemplates((prev) => [...prev, newTemplate])
      setSelectedTemplate(newTemplate)
      setIsCreateMode(false)
      setEditedTemplate({})
    }
  }

  // Handle deleting a template
  const handleDeleteTemplate = () => {
    if (selectedTemplate) {
      setTemplates((prev) => prev.filter((t) => t.id !== selectedTemplate.id))
      setSelectedTemplate(null)
    }
  }

  // Handle copying template code to clipboard
  const handleCopyCode = (code: string) => {
    navigator.clipboard.writeText(code)
  }

  // Get category icon
  const getCategoryIcon = (category: string) => {
    switch (category) {
      case "schema":
        return <Database className="h-4 w-4" />
      case "data_quality":
        return <FileCode className="h-4 w-4" />
      case "drift":
        return <AlertTriangle className="h-4 w-4" />
      case "performance":
        return <BarChart className="h-4 w-4" />
      case "pipeline":
        return <Clock className="h-4 w-4" />
      default:
        return <FileCode className="h-4 w-4" />
    }
  }

  return (
    <Card className="w-full h-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle>Fix Templates Library</CardTitle>
          <Button variant="outline" size="sm" onClick={handleCreateTemplate}>
            <Plus className="h-4 w-4 mr-1" />
            New Template
          </Button>
        </div>
        <CardDescription>Browse and apply predefined fixes for common issues</CardDescription>
      </CardHeader>

      <div className="px-6 pb-3">
        <div className="flex items-center space-x-2">
          <div className="relative flex-1">
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search templates..."
              className="pl-8"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <Button variant="outline" size="icon" title="Filter templates">
            <Filter className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center justify-between mt-3">
          <Tabs defaultValue="all" value={activeCategory} onValueChange={setActiveCategory} className="w-full">
            <TabsList className="grid grid-cols-7">
              <TabsTrigger value="all">All</TabsTrigger>
              <TabsTrigger value="schema">Schema</TabsTrigger>
              <TabsTrigger value="data_quality">Data Quality</TabsTrigger>
              <TabsTrigger value="drift">Drift</TabsTrigger>
              <TabsTrigger value="performance">Performance</TabsTrigger>
              <TabsTrigger value="pipeline">Pipeline</TabsTrigger>
              <TabsTrigger value="custom">Custom</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>

        <div className="flex items-center justify-between mt-3">
          <div className="flex items-center space-x-2">
            <Switch id="favorites-filter" checked={showFavoritesOnly} onCheckedChange={setShowFavoritesOnly} />
            <Label htmlFor="favorites-filter" className="text-sm cursor-pointer">
              Favorites only
            </Label>
          </div>
          <div className="flex items-center space-x-2">
            <Switch id="autofix-filter" checked={showAutoFixOnly} onCheckedChange={setShowAutoFixOnly} />
            <Label htmlFor="autofix-filter" className="text-sm cursor-pointer">
              Auto-fix compatible
            </Label>
          </div>
        </div>
      </div>

      <CardContent className="p-0">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-0 h-[calc(100vh-20rem)]">
          <div className="border-r">
            <ScrollArea className="h-[calc(100vh-20rem)]">
              <div className="p-4 space-y-2">
                {filteredTemplates.length > 0 ? (
                  filteredTemplates.map((template) => (
                    <div
                      key={template.id}
                      className={cn(
                        "p-3 rounded-md cursor-pointer hover:bg-muted transition-colors",
                        selectedTemplate?.id === template.id && "bg-muted",
                      )}
                      onClick={() => handleSelectTemplate(template)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          {getCategoryIcon(template.category)}
                          <span className="font-medium">{template.name}</span>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6"
                          onClick={(e) => {
                            e.stopPropagation()
                            handleToggleFavorite(template.id)
                          }}
                        >
                          {template.isFavorite ? (
                            <Star className="h-4 w-4 text-yellow-500" />
                          ) : (
                            <StarOff className="h-4 w-4 text-muted-foreground" />
                          )}
                        </Button>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1 line-clamp-2">{template.description}</p>
                      <div className="flex items-center justify-between mt-2">
                        <div className="flex flex-wrap gap-1">
                          {template.autoFixCompatible && (
                            <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-500">
                              <Zap className="h-3 w-3 mr-1" />
                              Auto-fix
                            </Badge>
                          )}
                          {template.tags.slice(0, 2).map((tag) => (
                            <Badge key={tag} variant="outline" className="text-xs">
                              {tag}
                            </Badge>
                          ))}
                          {template.tags.length > 2 && (
                            <Badge variant="outline" className="text-xs">
                              +{template.tags.length - 2}
                            </Badge>
                          )}
                        </div>
                        <span className="text-xs text-muted-foreground">Used {template.usageCount} times</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                    <p>No templates found</p>
                    <Button
                      variant="link"
                      onClick={() => {
                        setSearchQuery("")
                        setActiveCategory("all")
                        setShowFavoritesOnly(false)
                        setShowAutoFixOnly(false)
                      }}
                    >
                      Clear filters
                    </Button>
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>

          <div className="col-span-1 md:col-span-1 lg:col-span-2 border-t md:border-t-0">
            <ScrollArea className="h-[calc(100vh-20rem)]">
              {selectedTemplate ? (
                <div className="p-6">
                  {isEditMode || isCreateMode ? (
                    // Edit mode
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="template-name">Template Name</Label>
                        <Input
                          id="template-name"
                          value={editedTemplate.name || ""}
                          onChange={(e) => setEditedTemplate({ ...editedTemplate, name: e.target.value })}
                          className="mt-1"
                        />
                      </div>

                      <div>
                        <Label htmlFor="template-description">Description</Label>
                        <Textarea
                          id="template-description"
                          value={editedTemplate.description || ""}
                          onChange={(e) => setEditedTemplate({ ...editedTemplate, description: e.target.value })}
                          className="mt-1"
                          rows={3}
                        />
                      </div>

                      <div>
                        <Label htmlFor="template-category">Category</Label>
                        <Select
                          value={editedTemplate.category}
                          onValueChange={(value) => setEditedTemplate({ ...editedTemplate, category: value as any })}
                        >
                          <SelectTrigger id="template-category" className="mt-1">
                            <SelectValue placeholder="Select category" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectGroup>
                              <SelectItem value="schema">Schema</SelectItem>
                              <SelectItem value="data_quality">Data Quality</SelectItem>
                              <SelectItem value="drift">Drift</SelectItem>
                              <SelectItem value="performance">Performance</SelectItem>
                              <SelectItem value="pipeline">Pipeline</SelectItem>
                              <SelectItem value="custom">Custom</SelectItem>
                            </SelectGroup>
                          </SelectContent>
                        </Select>
                      </div>

                      <div>
                        <Label htmlFor="template-code">Fix Code</Label>
                        <Textarea
                          id="template-code"
                          value={editedTemplate.code || ""}
                          onChange={(e) => setEditedTemplate({ ...editedTemplate, code: e.target.value })}
                          className="mt-1 font-mono text-sm"
                          rows={10}
                        />
                      </div>

                      <div>
                        <Label htmlFor="template-tags">Tags (comma separated)</Label>
                        <Input
                          id="template-tags"
                          value={(editedTemplate.tags || []).join(", ")}
                          onChange={(e) =>
                            setEditedTemplate({
                              ...editedTemplate,
                              tags: e.target.value.split(",").map((tag) => tag.trim()),
                            })
                          }
                          className="mt-1"
                        />
                      </div>

                      <div className="flex items-center space-x-2">
                        <Switch
                          id="auto-fix-compatible"
                          checked={editedTemplate.autoFixCompatible || false}
                          onCheckedChange={(checked) =>
                            setEditedTemplate({ ...editedTemplate, autoFixCompatible: checked })
                          }
                        />
                        <Label htmlFor="auto-fix-compatible">Auto-fix compatible</Label>
                      </div>

                      {editedTemplate.autoFixCompatible && (
                        <div>
                          <Label htmlFor="confidence-threshold">
                            Confidence Threshold: {editedTemplate.confidenceThreshold || 80}%
                          </Label>
                          <input
                            id="confidence-threshold"
                            type="range"
                            min="50"
                            max="100"
                            step="5"
                            value={editedTemplate.confidenceThreshold || 80}
                            onChange={(e) =>
                              setEditedTemplate({
                                ...editedTemplate,
                                confidenceThreshold: Number.parseInt(e.target.value),
                              })
                            }
                            className="w-full mt-1"
                          />
                        </div>
                      )}

                      <div className="flex justify-end space-x-2 pt-4">
                        <Button
                          variant="outline"
                          onClick={() => {
                            setIsEditMode(false)
                            setIsCreateMode(false)
                            setEditedTemplate({})
                          }}
                        >
                          <X className="h-4 w-4 mr-1" />
                          Cancel
                        </Button>
                        <Button
                          onClick={isCreateMode ? handleSaveNewTemplate : handleSaveTemplate}
                          disabled={!editedTemplate.name || !editedTemplate.code}
                        >
                          <Save className="h-4 w-4 mr-1" />
                          Save Template
                        </Button>
                      </div>
                    </div>
                  ) : (
                    // View mode
                    <div>
                      <div className="flex items-center justify-between">
                        <h2 className="text-xl font-semibold">{selectedTemplate.name}</h2>
                        <div className="flex items-center space-x-1">
                          <Button variant="ghost" size="icon" onClick={() => handleToggleFavorite(selectedTemplate.id)}>
                            {selectedTemplate.isFavorite ? (
                              <Star className="h-4 w-4 text-yellow-500" />
                            ) : (
                              <StarOff className="h-4 w-4" />
                            )}
                          </Button>
                          <Button variant="ghost" size="icon" onClick={handleEditTemplate}>
                            <Edit className="h-4 w-4" />
                          </Button>
                          <Dialog>
                            <DialogTrigger asChild>
                              <Button variant="ghost" size="icon">
                                <Trash2 className="h-4 w-4 text-destructive" />
                              </Button>
                            </DialogTrigger>
                            <DialogContent>
                              <DialogHeader>
                                <DialogTitle>Delete Template</DialogTitle>
                                <DialogDescription>
                                  Are you sure you want to delete this template? This action cannot be undone.
                                </DialogDescription>
                              </DialogHeader>
                              <DialogFooter>
                                <Button variant="outline" onClick={() => {}}>
                                  Cancel
                                </Button>
                                <Button variant="destructive" onClick={handleDeleteTemplate}>
                                  Delete
                                </Button>
                              </DialogFooter>
                            </DialogContent>
                          </Dialog>
                        </div>
                      </div>

                      <p className="mt-2 text-muted-foreground">{selectedTemplate.description}</p>

                      <div className="flex flex-wrap gap-2 mt-4">
                        <Badge variant="outline" className="capitalize">
                          {getCategoryIcon(selectedTemplate.category)}
                          <span className="ml-1">{selectedTemplate.category.replace("_", " ")}</span>
                        </Badge>
                        {selectedTemplate.autoFixCompatible && (
                          <Badge variant="outline" className="bg-blue-500/10 text-blue-500">
                            <Zap className="h-3 w-3 mr-1" />
                            Auto-fix compatible ({selectedTemplate.confidenceThreshold}%)
                          </Badge>
                        )}
                        {selectedTemplate.tags.map((tag) => (
                          <Badge key={tag} variant="outline">
                            {tag}
                          </Badge>
                        ))}
                      </div>

                      <div className="mt-6">
                        <div className="flex items-center justify-between">
                          <h3 className="text-sm font-medium">Fix Code</h3>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8"
                            onClick={() => handleCopyCode(selectedTemplate.code)}
                          >
                            <Copy className="h-3 w-3 mr-1" />
                            Copy
                          </Button>
                        </div>
                        <pre className="mt-2 p-4 bg-muted rounded-md overflow-x-auto font-mono text-sm">
                          <code>{selectedTemplate.code}</code>
                        </pre>
                      </div>

                      <div className="flex items-center justify-between mt-6 text-sm text-muted-foreground">
                        <div>
                          Created by {selectedTemplate.author} on {selectedTemplate.createdAt.toLocaleDateString()}
                        </div>
                        <div>Used {selectedTemplate.usageCount} times</div>
                      </div>

                      <div className="flex justify-end mt-6">
                        <Button onClick={handleApplyTemplate}>Apply Template</Button>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground p-6">
                  {isCreateMode ? (
                    <div className="space-y-4 w-full max-w-2xl">
                      <h2 className="text-xl font-semibold">Create New Template</h2>
                      <div>
                        <Label htmlFor="new-template-name">Template Name</Label>
                        <Input
                          id="new-template-name"
                          value={editedTemplate.name || ""}
                          onChange={(e) => setEditedTemplate({ ...editedTemplate, name: e.target.value })}
                          className="mt-1"
                          placeholder="Enter template name"
                        />
                      </div>

                      <div>
                        <Label htmlFor="new-template-description">Description</Label>
                        <Textarea
                          id="new-template-description"
                          value={editedTemplate.description || ""}
                          onChange={(e) => setEditedTemplate({ ...editedTemplate, description: e.target.value })}
                          className="mt-1"
                          placeholder="Describe what this template fixes"
                          rows={3}
                        />
                      </div>

                      <div>
                        <Label htmlFor="new-template-category">Category</Label>
                        <Select
                          value={editedTemplate.category as string}
                          onValueChange={(value) => setEditedTemplate({ ...editedTemplate, category: value as any })}
                        >
                          <SelectTrigger id="new-template-category" className="mt-1">
                            <SelectValue placeholder="Select category" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectGroup>
                              <SelectItem value="schema">Schema</SelectItem>
                              <SelectItem value="data_quality">Data Quality</SelectItem>
                              <SelectItem value="drift">Drift</SelectItem>
                              <SelectItem value="performance">Performance</SelectItem>
                              <SelectItem value="pipeline">Pipeline</SelectItem>
                              <SelectItem value="custom">Custom</SelectItem>
                            </SelectGroup>
                          </SelectContent>
                        </Select>
                      </div>

                      <div>
                        <Label htmlFor="new-template-code">Fix Code</Label>
                        <Textarea
                          id="new-template-code"
                          value={editedTemplate.code || ""}
                          onChange={(e) => setEditedTemplate({ ...editedTemplate, code: e.target.value })}
                          className="mt-1 font-mono text-sm"
                          placeholder="Enter the code that fixes the issue"
                          rows={10}
                        />
                      </div>

                      <div>
                        <Label htmlFor="new-template-tags">Tags (comma separated)</Label>
                        <Input
                          id="new-template-tags"
                          value={(editedTemplate.tags || []).join(", ")}
                          onChange={(e) =>
                            setEditedTemplate({
                              ...editedTemplate,
                              tags: e.target.value.split(",").map((tag) => tag.trim()),
                            })
                          }
                          className="mt-1"
                          placeholder="schema, type-conversion, etc."
                        />
                      </div>

                      <div className="flex items-center space-x-2">
                        <Switch
                          id="new-auto-fix-compatible"
                          checked={editedTemplate.autoFixCompatible || false}
                          onCheckedChange={(checked) =>
                            setEditedTemplate({ ...editedTemplate, autoFixCompatible: checked })
                          }
                        />
                        <Label htmlFor="new-auto-fix-compatible">Auto-fix compatible</Label>
                      </div>

                      {editedTemplate.autoFixCompatible && (
                        <div>
                          <Label htmlFor="new-confidence-threshold">
                            Confidence Threshold: {editedTemplate.confidenceThreshold || 80}%
                          </Label>
                          <input
                            id="new-confidence-threshold"
                            type="range"
                            min="50"
                            max="100"
                            step="5"
                            value={editedTemplate.confidenceThreshold || 80}
                            onChange={(e) =>
                              setEditedTemplate({
                                ...editedTemplate,
                                confidenceThreshold: Number.parseInt(e.target.value),
                              })
                            }
                            className="w-full mt-1"
                          />
                        </div>
                      )}

                      <div className="flex justify-end space-x-2 pt-4">
                        <Button
                          variant="outline"
                          onClick={() => {
                            setIsCreateMode(false)
                            setEditedTemplate({})
                          }}
                        >
                          <X className="h-4 w-4 mr-1" />
                          Cancel
                        </Button>
                        <Button onClick={handleSaveNewTemplate} disabled={!editedTemplate.name || !editedTemplate.code}>
                          <Save className="h-4 w-4 mr-1" />
                          Create Template
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <FileCode className="h-16 w-16 mb-4 opacity-50" />
                      <p>Select a template to view details</p>
                      <p className="text-sm mt-2">
                        or{" "}
                        <Button variant="link" className="p-0 h-auto" onClick={handleCreateTemplate}>
                          create a new template
                        </Button>
                      </p>
                    </>
                  )}
                </div>
              )}
            </ScrollArea>
          </div>
        </div>
      </CardContent>

      <CardFooter className="flex justify-between border-t p-4">
        <div className="text-sm text-muted-foreground">
          {filteredTemplates.length} of {templates.length} templates
        </div>
        {onClose && (
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
        )}
      </CardFooter>
    </Card>
  )
}
