"use client"

import { useState } from "react"
import { Check, ChevronsUpDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface ModelFilterProps {
  models: { id: string; name: string; type: string }[]
  selectedModels: string[]
  onSelectionChange: (selectedIds: string[]) => void
  className?: string
}

export function ModelFilter({ models, selectedModels, onSelectionChange, className }: ModelFilterProps) {
  const [open, setOpen] = useState(false)

  const toggleModel = (modelId: string) => {
    if (selectedModels.includes(modelId)) {
      onSelectionChange(selectedModels.filter((id) => id !== modelId))
    } else {
      onSelectionChange([...selectedModels, modelId])
    }
  }

  const modelTypes = Array.from(new Set(models.map((model) => model.type)))

  return (
    <div className={className}>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button variant="outline" role="combobox" aria-expanded={open} className="justify-between">
            {selectedModels.length > 0 ? (
              <span className="flex items-center gap-1">
                <span>Selected Models</span>
                <Badge className="ml-1 rounded-full">{selectedModels.length}</Badge>
              </span>
            ) : (
              "Select Models"
            )}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[300px] p-0">
          <Command>
            <CommandInput placeholder="Search models..." />
            <CommandList>
              <CommandEmpty>No models found.</CommandEmpty>
              {modelTypes.map((type) => (
                <CommandGroup key={type} heading={type}>
                  {models
                    .filter((model) => model.type === type)
                    .map((model) => (
                      <CommandItem key={model.id} value={model.id} onSelect={() => toggleModel(model.id)}>
                        <Check
                          className={cn(
                            "mr-2 h-4 w-4",
                            selectedModels.includes(model.id) ? "opacity-100" : "opacity-0",
                          )}
                        />
                        {model.name}
                      </CommandItem>
                    ))}
                </CommandGroup>
              ))}
              <CommandSeparator />
              <CommandGroup>
                <CommandItem onSelect={() => onSelectionChange([])} className="justify-center text-center">
                  Clear selection
                </CommandItem>
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  )
}
