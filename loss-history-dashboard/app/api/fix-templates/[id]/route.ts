import { NextResponse } from 'next/server';
import { prisma } from '@/lib/db';

// GET /api/fix-templates/[id] - Get a specific template
export async function GET(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    const template = await prisma.fixTemplate.findUnique({
      where: { id: params.id }
    });

    if (!template) {
      return NextResponse.json(
        { success: false, error: 'Template not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({ success: true, data: template });
  } catch (error) {
    console.error(`Error fetching template ${params.id}:`, error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch template' },
      { status: 500 }
    );
  }
}

// PUT /api/fix-templates/[id] - Update a template
export async function PUT(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    const body = await request.json();
    const { name, description, problemType, solution, code } = body;

    // Basic validation
    if (!name || !problemType || !solution) {
      return NextResponse.json(
        { success: false, error: 'Missing required fields' },
        { status: 400 }
      );
    }

    // Check if template exists
    const existingTemplate = await prisma.fixTemplate.findUnique({
      where: { id: params.id }
    });

    if (!existingTemplate) {
      return NextResponse.json(
        { success: false, error: 'Template not found' },
        { status: 404 }
      );
    }

    // Update template
    const updatedTemplate = await prisma.fixTemplate.update({
      where: { id: params.id },
      data: {
        name,
        description: description || '',
        problemType,
        solution,
        code: code || '',
      }
    });

    return NextResponse.json({ success: true, data: updatedTemplate });
  } catch (error) {
    console.error(`Error updating template ${params.id}:`, error);
    return NextResponse.json(
      { success: false, error: 'Failed to update template' },
      { status: 500 }
    );
  }
}

// DELETE /api/fix-templates/[id] - Delete a template
export async function DELETE(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    // Check if template exists
    const existingTemplate = await prisma.fixTemplate.findUnique({
      where: { id: params.id }
    });

    if (!existingTemplate) {
      return NextResponse.json(
        { success: false, error: 'Template not found' },
        { status: 404 }
      );
    }

    // Delete template
    await prisma.fixTemplate.delete({
      where: { id: params.id }
    });

    return NextResponse.json({ 
      success: true, 
      message: 'Template deleted successfully' 
    });
  } catch (error) {
    console.error(`Error deleting template ${params.id}:`, error);
    return NextResponse.json(
      { success: false, error: 'Failed to delete template' },
      { status: 500 }
    );
  }
} 